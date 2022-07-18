use core::fmt;
use core::iter::FusedIterator;
use core::mem::MaybeUninit;
use core::ops::{Deref, Range};
use core::ptr;
use core::slice;

/// A "history buffer", similar to a write-only ring buffer of fixed length.
///
/// This buffer keeps a fixed number of elements.  On write, the oldest element
/// is overwritten. Thus, the buffer is useful to keep a history of values with
/// some desired depth, and for example calculate a rolling average.
///
/// # Examples
/// ```
/// use heapless::HistoryBuffer;
///
/// // Initialize a new buffer with 8 elements.
/// let mut buf = HistoryBuffer::<_, 8>::new();
///
/// // Starts with no data
/// assert_eq!(buf.recent(), None);
///
/// buf.write(3);
/// buf.write(5);
/// buf.extend(&[4, 4]);
///
/// // The most recent written element is a four.
/// assert_eq!(buf.recent(), Some(&4));
///
/// // To access all elements in an unspecified order, use `as_slice()`.
/// for el in buf.as_slice() { println!("{:?}", el); }
///
/// // Now we can prepare an average of all values, which comes out to 4.
/// let avg = buf.as_slice().iter().sum::<usize>() / buf.len();
/// assert_eq!(avg, 4);
/// ```
pub struct HistoryBuffer<T, const N: usize> {
    data: [MaybeUninit<T>; N],
    write_at: usize,
    filled: bool,
}

impl<T, const N: usize> HistoryBuffer<T, N> {
    const INIT: MaybeUninit<T> = MaybeUninit::uninit();

    /// Constructs a new history buffer.
    ///
    /// The construction of a `HistoryBuffer` works in `const` contexts.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::HistoryBuffer;
    ///
    /// // Allocate a 16-element buffer on the stack
    /// let x: HistoryBuffer<u8, 16> = HistoryBuffer::new();
    /// assert_eq!(x.len(), 0);
    /// ```
    #[inline]
    pub const fn new() -> Self {
        // Const assert
        crate::sealed::greater_than_0::<N>();

        Self {
            data: [Self::INIT; N],
            write_at: 0,
            filled: false,
        }
    }

    /// Clears the buffer, replacing every element with the default value of
    /// type `T`.
    pub fn clear(&mut self) {
        *self = Self::new();
    }
}

impl<T, const N: usize> HistoryBuffer<T, N>
where
    T: Copy + Clone,
{
    /// Constructs a new history buffer, where every element is the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::HistoryBuffer;
    ///
    /// // Allocate a 16-element buffer on the stack
    /// let mut x: HistoryBuffer<u8, 16> = HistoryBuffer::new_with(4);
    /// // All elements are four
    /// assert_eq!(x.as_slice(), [4; 16]);
    /// ```
    #[inline]
    pub fn new_with(t: T) -> Self {
        Self {
            data: [MaybeUninit::new(t); N],
            write_at: 0,
            filled: true,
        }
    }

    /// Clears the buffer, replacing every element with the given value.
    pub fn clear_with(&mut self, t: T) {
        *self = Self::new_with(t);
    }
}

impl<T, const N: usize> HistoryBuffer<T, N> {
    /// Returns the current fill level of the buffer.
    #[inline]
    pub const fn len(&self) -> usize {
        if self.filled {
            N
        } else {
            self.write_at
        }
    }

    /// Returns the capacity of the buffer, which is the length of the
    /// underlying backing array.
    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Writes an element to the buffer, overwriting the oldest value.
    pub fn write(&mut self, t: T) {
        if self.filled {
            // Drop the old before we overwrite it.
            unsafe { ptr::drop_in_place(self.data[self.write_at].as_mut_ptr()) }
        }
        self.data[self.write_at] = MaybeUninit::new(t);

        self.write_at += 1;
        if self.write_at == self.capacity() {
            self.write_at = 0;
            self.filled = true;
        }
    }

    /// Clones and writes all elements in a slice to the buffer.
    ///
    /// If the slice is longer than the buffer, only the last
    /// [`self.len`](HistoryBuffer#method.len) elements will actually be stored.
    pub fn extend_from_slice(&mut self, other: &[T])
    where
        T: Clone,
    {
        for item in other {
            self.write(item.clone());
        }
    }

    /// Returns a reference to the most recently written value.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::HistoryBuffer;
    ///
    /// let mut x: HistoryBuffer<u8, 16> = HistoryBuffer::new();
    /// x.write(4);
    /// x.write(10);
    /// assert_eq!(x.recent(), Some(&10));
    /// ```
    pub const fn recent(&self) -> Option<&T> {
        match self.most_recent_index() {
            Some(i) => Some(unsafe { &*self.data[i].as_ptr() }),
            None => None,
        }
    }

    /// Returns the index of the most recently written value.
    ///
    /// If no values have been written to the buffer, returns `None`.
    ///
    /// This is intended to be used for low-level access to the buffer contents
    /// together with [`as_slice`](HistoryBuffer#method.as_slice). If you just
    /// want to iterate over the values in the buffer, use
    /// [`oldest_ordered`](HistoryBuffer#method.oldest_ordered), possibly along
    /// with [`rev`](Iterator#method.rev).
    ///
    /// See also [`oldest_index`](HistoryBuffer#method.oldest_index).
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::HistoryBuffer;
    ///
    /// let mut buf: HistoryBuffer<u8, 10> = HistoryBuffer::new();
    /// assert_eq!(buf.most_recent_index(), None);
    /// buf.write(0);
    /// assert_eq!(buf.most_recent_index(), Some(0));
    /// buf.write(1);
    /// assert_eq!(buf.most_recent_index(), Some(1));
    /// ```
    pub const fn most_recent_index(&self) -> Option<usize> {
        if self.write_at == 0 {
            if self.filled {
                Some(self.capacity() - 1)
            } else {
                None
            }
        } else {
            Some(self.write_at - 1)
        }
    }

    /// Returns a reference to the oldest value.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::HistoryBuffer;
    ///
    /// let mut x: HistoryBuffer<u8, 16> = HistoryBuffer::new();
    /// x.write(4);
    /// x.write(10);
    /// assert_eq!(x.oldest(), Some(&4));
    /// ```
    pub const fn oldest(&self) -> Option<&T> {
        self.nth_oldest(0)
    }

    /// Returns a reference to the `n`th oldest value in the buffer.
    ///
    /// Returns `None` when `n >= N`.
    ///
    /// This is intended for reading specific values from the buffer. If you
    /// just want to iterate over *all* of the values in the buffer, use
    /// [`oldest_ordered`](HistoryBuffer#method.oldest_ordered).
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::HistoryBuffer;
    ///
    /// let mut x: HistoryBuffer<u8, 16> = HistoryBuffer::new();
    /// x.write(4);
    /// x.write(10);
    /// assert_eq!(x.nth_oldest(0), Some(&4));
    /// assert_eq!(x.nth_oldest(1), Some(&10));
    /// assert_eq!(x.nth_oldest(2), None);
    /// ```
    pub const fn nth_oldest(&self, mut n: usize) -> Option<&T> {
        if n >= self.len() {
            return None;
        }

        if self.filled {
            let (sum, overflowed) = n.overflowing_add(self.write_at);
            n = sum;

            if n >= self.len() {
                n -= self.len();
            }

            if overflowed {
                n += usize::MAX - self.len() + 1;
            }
        } else if self.write_at == 0 {
            return None; // Buffer is empty
        }

        Some(unsafe { &*self.data[n].as_ptr() })
    }

    /// Returns the index of the oldest value in the buffer.
    ///
    /// If no values have been written to the buffer, returns `None`.
    ///
    /// This is intended to be used for low-level access to the buffer contents
    /// together with [`as_slice`](HistoryBuffer#method.as_slice). If you just
    /// want to iterate over the values in the buffer, use
    /// [`oldest_ordered`](HistoryBuffer#method.oldest_ordered).
    ///
    /// See also [`most_recent_index`](HistoryBuffer#method.most_recent_index).
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::HistoryBuffer;
    ///
    /// let mut buf: HistoryBuffer<u8, 10> = HistoryBuffer::new();
    /// assert_eq!(buf.oldest_index(), None);
    /// buf.write(0);
    /// assert_eq!(buf.oldest_index(), Some(0));
    /// buf.write(1);
    /// assert_eq!(buf.oldest_index(), Some(0));
    /// buf.extend_from_slice(&[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    /// assert_eq!(buf.oldest_index(), Some(2));
    /// ```
    pub const fn oldest_index(&self) -> Option<usize> {
        if !self.filled {
            if self.write_at == 0 {
                None
            } else {
                Some(0)
            }
        } else {
            Some(self.write_at)
        }
    }

    /// Returns the array slice backing the buffer, without keeping track
    /// of the write position. Therefore, the element order is unspecified.
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.data.as_ptr() as *const _, self.len()) }
    }

    /// Returns an iterator for iterating over the buffer from oldest to newest.
    ///
    /// # Examples
    ///
    /// ```
    /// use heapless::HistoryBuffer;
    ///
    /// let mut buffer: HistoryBuffer<u8, 6> = HistoryBuffer::new();
    /// buffer.extend([0, 0, 0, 1, 2, 3, 4, 5, 6]);
    /// let expected = [1, 2, 3, 4, 5, 6];
    /// for (x, y) in buffer.oldest_ordered().zip(expected.iter()) {
    ///     assert_eq!(x, y)
    /// }
    /// ```
    pub const fn oldest_ordered<'a>(&'a self) -> OldestOrdered<'a, T, N> {
        OldestOrdered {
            buf: self,
            idxs: 0..self.len(),
        }
    }
}

impl<T, const N: usize> Extend<T> for HistoryBuffer<T, N> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for item in iter.into_iter() {
            self.write(item);
        }
    }
}

impl<'a, T, const N: usize> Extend<&'a T> for HistoryBuffer<T, N>
where
    T: 'a + Clone,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = &'a T>,
    {
        self.extend(iter.into_iter().cloned())
    }
}

impl<T, const N: usize> Drop for HistoryBuffer<T, N> {
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut T,
                self.len(),
            ))
        }
    }
}

impl<T, const N: usize> Deref for HistoryBuffer<T, N> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const N: usize> AsRef<[T]> for HistoryBuffer<T, N> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<T, const N: usize> fmt::Debug for HistoryBuffer<T, N>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <[T] as fmt::Debug>::fmt(self, f)
    }
}

impl<T, const N: usize> Default for HistoryBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// An iterator on the underlying buffer ordered from oldest data to newest.
///
/// Created by
/// [`HistoryBuffer.oldest_ordered`](HistoryBuffer#method.oldest_ordered).
#[derive(Clone)]
pub struct OldestOrdered<'a, T, const N: usize> {
    buf: &'a HistoryBuffer<T, N>,
    idxs: Range<usize>,
}

impl<'a, T, const N: usize> Iterator for OldestOrdered<'a, T, N> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        self.idxs.next().and_then(|n| self.buf.nth_oldest(n))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for OldestOrdered<'a, T, N> {
    fn next_back(&mut self) -> Option<&'a T> {
        self.idxs.next_back().and_then(|n| self.buf.nth_oldest(n))
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for OldestOrdered<'a, T, N> {
    fn len(&self) -> usize {
        self.idxs.len()
    }
}

impl<'a, T, const N: usize> FusedIterator for OldestOrdered<'a, T, N> {}

#[cfg(test)]
mod tests {
    use crate::HistoryBuffer;
    use core::fmt::Debug;

    #[test]
    fn new() {
        let x: HistoryBuffer<u8, 4> = HistoryBuffer::new_with(1);
        assert_eq!(x.len(), 4);
        assert_eq!(x.as_slice(), [1; 4]);
        assert_eq!(*x, [1; 4]);

        let x: HistoryBuffer<u8, 4> = HistoryBuffer::new();
        assert_eq!(x.as_slice(), []);
    }

    #[test]
    fn write() {
        let mut x: HistoryBuffer<u8, 4> = HistoryBuffer::new();
        x.write(1);
        x.write(4);
        assert_eq!(x.as_slice(), [1, 4]);

        x.write(5);
        x.write(6);
        x.write(10);
        assert_eq!(x.as_slice(), [10, 4, 5, 6]);

        x.extend([11, 12].iter());
        assert_eq!(x.as_slice(), [10, 11, 12, 6]);
    }

    #[test]
    fn clear() {
        let mut x: HistoryBuffer<u8, 4> = HistoryBuffer::new_with(1);
        x.clear();
        assert_eq!(x.as_slice(), []);

        let mut x: HistoryBuffer<u8, 4> = HistoryBuffer::new();
        x.clear_with(1);
        assert_eq!(x.as_slice(), [1; 4]);
    }

    #[test]
    fn recent_and_oldest() {
        let mut x: HistoryBuffer<u8, 4> = HistoryBuffer::new();
        assert_eq!(x.recent(), None);
        assert_eq!(x.most_recent_index(), None);
        assert_eq!(x.oldest_index(), None);
        assert_eq!(x.oldest(), None);

        x.write(1);
        x.write(4);
        assert_eq!(x.recent(), Some(&4));
        assert_eq!(x.most_recent_index(), Some(1));
        assert_eq!(x.oldest_index(), Some(0));
        assert_eq!(x.oldest(), Some(&1));

        x.write(5);
        x.write(6);
        x.write(10);
        assert_eq!(x.recent(), Some(&10));
        assert_eq!(x.most_recent_index(), Some(0));
        assert_eq!(x.oldest_index(), Some(1));
        assert_eq!(x.oldest(), Some(&4));
    }

    #[test]
    fn nth_oldest() {
        let mut x: HistoryBuffer<u8, 4> = HistoryBuffer::new();
        assert_eq!(x.nth_oldest(0), None);
        assert_eq!(x.nth_oldest(1), None);
        assert_eq!(x.nth_oldest(2), None);
        assert_eq!(x.nth_oldest(3), None);
        assert_eq!(x.nth_oldest(4), None);

        x.write(1);
        x.write(4);

        assert_eq!(x.nth_oldest(0), Some(&1));
        assert_eq!(x.nth_oldest(1), Some(&4));
        assert_eq!(x.nth_oldest(2), None);
        assert_eq!(x.nth_oldest(3), None);
        assert_eq!(x.nth_oldest(4), None);

        x.write(5);
        x.write(6);
        x.write(10);

        assert_eq!(x.nth_oldest(0), Some(&4));
        assert_eq!(x.nth_oldest(1), Some(&5));
        assert_eq!(x.nth_oldest(2), Some(&6));
        assert_eq!(x.nth_oldest(3), Some(&10));
        assert_eq!(x.nth_oldest(4), None);
    }

    #[test]
    fn as_slice() {
        let mut x: HistoryBuffer<u8, 4> = HistoryBuffer::new();

        assert_eq!(x.as_slice(), []);

        x.extend([1, 2, 3, 4, 5].iter());

        assert_eq!(x.as_slice(), [5, 2, 3, 4]);
    }

    #[test]
    fn ordered_empty() {
        // test on an empty buffer
        let buffer: HistoryBuffer<u8, 6> = HistoryBuffer::new();
        let mut iter = buffer.oldest_ordered();
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn ordered_non_filled() {
        // test on a un-filled buffer
        let mut buffer: HistoryBuffer<u8, 6> = HistoryBuffer::new();
        buffer.extend([1, 2, 3]);
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.oldest_ordered().len(), 3);
        assert_eq_iter(buffer.oldest_ordered(), &[1, 2, 3]);
        assert_eq_iter_rev(buffer.oldest_ordered(), &[3, 2, 1]);
    }

    #[test]
    fn ordered_filled() {
        // test on a filled buffer
        let mut buffer: HistoryBuffer<u8, 6> = HistoryBuffer::new();
        buffer.extend([0, 0, 0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(buffer.len(), 6);
        assert_eq!(buffer.oldest_ordered().len(), 6);
        assert_eq_iter(buffer.oldest_ordered(), &[1, 2, 3, 4, 5, 6]);
        assert_eq_iter_rev(buffer.oldest_ordered(), &[6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn ordered_comprehensive() {
        // comprehensive test all cases
        for n in 0..50 {
            const N: usize = 7;
            let mut buffer: HistoryBuffer<u8, N> = HistoryBuffer::new();
            buffer.extend(0..n);
            let range = n.saturating_sub(N as u8)..n;
            assert_eq!(buffer.oldest_ordered().len(), buffer.len());
            assert_eq_iter(buffer.oldest_ordered().copied(), range.clone());
            assert_eq_iter_rev(buffer.oldest_ordered().copied(), range.rev());
        }
    }

    /// Compares two iterators item by item, making sure they stop at the same time.
    fn assert_eq_iter<I: Eq + Debug>(
        a: impl IntoIterator<Item = I>,
        b: impl IntoIterator<Item = I>,
    ) {
        let mut a = a.into_iter();
        let mut b = b.into_iter();

        let mut i = 0;
        loop {
            let a_item = a.next();
            let b_item = b.next();

            assert_eq!(a_item, b_item, "{}", i);

            i += 1;

            if b_item.is_none() {
                break;
            }
        }
    }

    /// Compares two double ended iterators item by item in reverse, making sure
    /// they stop at the same time.
    fn assert_eq_iter_rev<A, B, I: Eq + Debug>(a: A, b: B)
    where
        A: IntoIterator<Item = I>,
        B: IntoIterator<Item = I>,
        <A as IntoIterator>::IntoIter: DoubleEndedIterator,
    {
        let mut a = a.into_iter();
        let mut b = b.into_iter();

        let mut i = 0;
        loop {
            let a_item = a.next_back();
            let b_item = b.next();

            assert_eq!(a_item, b_item, "{}", i);

            i += 1;

            if b_item.is_none() {
                break;
            }
        }
    }
}
