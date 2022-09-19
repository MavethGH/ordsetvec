use iter_set::{classify_by, ClassifyBy};
use std::{iter::Peekable, ops::Deref};
use thiserror::Error;

pub use iter_set::Inclusion;
pub use std::cmp::Ordering;

/// Must be implemented by all types in an `OrdSetVec`
/// Implemented for all types that are Ord, but implementors
/// do not need to be Ord, since they can also be compared
/// using a Key.
///
/// # Examples
///
/// Comparing by value:
/// ```
/// #[derive(PartialEq, Eq, PartialOrd, Ord)]
/// struct Item(u32);
///
/// impl OrdSetItemTrait for Item {
///     type Key = Item;
///
///     fn compare(a: &Self, b: &Self) -> Ordering {
///         a.cmp(b)
///     }
///
///     fn compare_key(a: &Self, b: &Self::Key) -> Ordering {
///         a.cmp(b)
///     }
/// }
/// ```
///
/// Comparing by Key:
/// ```
/// struct Item {
///     // The key used for comparisons
///     pub key: u32,
///     //Something that is expensive or inconvenient to compare to itself
///     data: Vec<String>,
/// }
///
/// impl OrdSetItemTrait for Item {
///     type Key = u32;
///
///     fn compare(a: &Self, b: &Self) -> Ordering {
///         a.key.cmp(&b.key)
///     }
///
///     fn compare_key(a: &Self, b: &Self::Key) -> Ordering {
///         a.key.cmp(b)
///     }
/// }
/// ```
pub trait OrdSetItemTrait {
    type Key;
    fn compare(a: &Self, b: &Self) -> Ordering;
    fn compare_key(a: &Self, b: &Self::Key) -> Ordering;
}

impl<T: Ord> OrdSetItemTrait for T {
    type Key = T;

    fn compare(a: &Self, b: &Self) -> Ordering {
        a.cmp(b)
    }

    fn compare_key(a: &Self, b: &Self::Key) -> Ordering {
        a.cmp(b)
    }
}

/// Errors that can occur when creating or verifying `OrdSetIter` or `OrdSetVec`
#[derive(Error, Debug)]
pub enum VerificationError {
    #[error("duplicate data")]
    DuplicateData,
    #[error("unsorted or duplicate data")]
    UnsortedOrDuplicate,
    #[error("item must compare greater than the previous item")]
    ItemTooSmall,
    #[error("item must compare less than the previous item")]
    ItemTooBig,
    #[error("key or item not found")]
    NotFound,
}

pub type Result<T> = std::result::Result<T, VerificationError>;

/// A collection of *unique, sorted* values with very fast iteration
/// (as fast as [`Vec`], cache-friendly), O(log(n)) lookup using binary search,
/// and poor insert/delete performance (O(n), just like a `Vec`)
pub struct OrdSetVec<T: OrdSetItemTrait> {
    inner: Vec<T>,
}

impl<T: OrdSetItemTrait> Default for OrdSetVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: OrdSetItemTrait> Deref for OrdSetVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: OrdSetItemTrait> OrdSetVec<T> {
    /// Creates a new, empty `OrdSetVec`
    ///
    /// See [`Vec::new()`]
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    /// Constructs a new, empty `OrdSetVec` with at least the specified capacity.
    ///
    /// See [`Vec::with_capacity()`]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Vec::with_capacity(capacity),
        }
    }

    /// Creates an `OrdSetVec` from an already sorted vector, verifying that it is indeed
    /// sorted and that there are no duplicate items.
    ///
    /// Returns an error if the Vec is not sorted or contains duplicates.
    pub fn from_sorted_vec(external: Vec<T>) -> Result<Self> {
        for i in 1..external.len() {
            match T::compare(&external[i - 1], &external[i]) {
                Ordering::Less => continue,
                Ordering::Equal => return Err(VerificationError::DuplicateData),
                Ordering::Greater => return Err(VerificationError::UnsortedOrDuplicate),
            }
        }

        Ok(Self::from_sorted_vec_unchecked(external))
    }

    /// Creates an `OrdSetVec` from an already sorted vector without any safeguards
    pub fn from_sorted_vec_unchecked(external: Vec<T>) -> Self {
        Self { inner: external }
    }

    /// Creates an `OrdSetVec` from an unsorted iterator
    ///
    /// Currently collects the iterator into a `Vec` and passes it to
    /// [`from_unsorted_vec()`](`Self::from_unsorted_vec()`)
    ///
    pub fn from_unsorted<I: Iterator<Item = T>>(iter: I) -> Result<Self> {
        Self::from_unsorted_vec(iter.collect())
    }

    /// Creates an `OrdSetVec` from an unsorted `Vec`.
    ///
    /// Use this
    ///
    /// Returns [`VerificationError::DuplicateData`] if the `Vec` contains duplicates
    pub fn from_unsorted_vec(mut external: Vec<T>) -> Result<Self> {
        external.sort_unstable_by(T::compare);

        match Self::find_dup(&external) {
            None => Ok(Self::from_sorted_vec_unchecked(external)),
            Some(_) => Err(VerificationError::DuplicateData),
        }
    }

    /// Appends an element to the back of the `OrdSetVec`.
    ///
    /// Returns an error if the item doesn't compare greater than the previous end item
    pub fn push(&mut self, item: T) -> Result<()> {
        if self.inner.is_empty() {
            self.inner.push(item);
        } else {
            match T::compare(&item, self.inner.last().unwrap()) {
                Ordering::Less => return Err(VerificationError::ItemTooSmall),
                Ordering::Equal => return Err(VerificationError::DuplicateData),
                Ordering::Greater => self.inner.push(item),
            }
        }

        Ok(())
    }

    /// Most `Vec` methods that make sense to use on an `OrdSetVec` are already
    /// implemented on it.
    ///
    /// Use this if you need to pass an `OrdSetVec` to something that needs a `Vec`
    pub fn as_vec(&self) -> &Vec<T> {
        &self.inner
    }

    /// Same as [`Self::as_vec()`], but returns a mutable reference.
    ///
    /// Use with caution, since mutating the resulting `Vec` can easily break
    /// the `OrdSetVec` guarantees of being sorted and deduplicated. If you
    /// need to use this but can't prevent that, use [`Self::verify()`] before
    /// further using this `OrdSetVec`, to ensure that it is valid.
    pub fn as_vec_mut_unchecked(&mut self) -> &mut Vec<T> {
        &mut self.inner
    }

    /// Converts this `OrdSetVec` into a normal `Vec`, consuming it.
    ///
    /// Use this if you no longer need the guarantees of `OrdSetVec` and just want your
    /// data as a normal `Vec`.
    pub fn into_vec(self) -> Vec<T> {
        self.inner
    }

    /// Sorts this `OrdSetVec` and removes any duplicates. Call this after using any
    /// `..._unchecked` method if it possibly invalidated the contents of this struct
    pub fn verify(&mut self) {
        self.inner.sort_unstable_by(T::compare);
        self.inner
            .dedup_by(|a, b| T::compare(a, b) == Ordering::Equal);
    }

    /// Return the index of an item in this `OrdSetVec` if it exists, by binary search.
    ///
    /// See [`slice::binary_search()`] for more details.
    pub fn binary_search_item(&self, item: &T) -> std::result::Result<usize, usize> {
        self.inner.binary_search_by(|e| T::compare(e, item))
    }

    /// Returns true if the item exists in this `OrdSetVec`, and false if it doesn't.
    ///
    /// Uses [`Self::binary_search_item()`] internally.
    pub fn contains_item(&self, item: &T) -> bool {
        self.binary_search_item(item).is_ok()
    }

    /// Returns the index of a key in this `OrdSetVec` if it exists, by binary search.
    ///
    /// See [`slice::binary_search_by()`] for details on the binary search, and [`OrdSetItemTrait`]
    /// for details on comparing by key.
    pub fn binary_search_key(&self, key: &T::Key) -> std::result::Result<usize, usize> {
        self.inner.binary_search_by(|e| T::compare_key(e, key))
    }

    /// Returns true if the key exists in this set, and false if it doesn't/
    ///
    /// uses [`Self::binary_search_key()`] internally.
    pub fn contains_key(&self, key: &T::Key) -> bool {
        self.binary_search_key(key).is_ok()
    }

    /// See [`Vec::len()`].
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// See [`Vec::is_empty()`].
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// See [`Vec::as_slice()`]
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Inserts an item into the `OrdSetVec`.
    ///
    /// Uses binary search to find the index to insert at, and returns that
    /// index if successful. Uses [`Vec::insert()`], which can be very slow
    /// when the size of the vector in bytes is large.
    ///
    /// Returns an error if the item is already present.
    pub fn insert(&mut self, item: T) -> Result<usize> {
        let index = match self.binary_search_item(&item) {
            Ok(_) => return Err(VerificationError::DuplicateData),
            Err(index) => index,
        };
        self.inner.insert(index, item);
        Ok(index)
    }

    /// Gets an item from the `OrdSetVec`, using a key.
    ///
    /// Returns None if the key doesn't match any value in the set
    pub fn by_key(&self, key: &T::Key) -> Option<&T> {
        match self.binary_search_key(key) {
            Ok(v) => Some(&self.inner[v]),
            Err(_) => None,
        }
    }

    /// Gets a mutable reference to an item in the `OrdSetVec`, using a key.
    ///
    /// Callers must either ensure that the returned reference isn't mutated in
    /// a way that messes up the ordering of the set, or call [`Self::verify()`]
    /// after every call to this (or any other `..._unchecked`) method.
    pub fn by_key_mut_unchecked(&mut self, key: &T::Key) -> Option<&mut T> {
        match self.binary_search_key(key) {
            Ok(v) => Some(&mut self.inner[v]),
            Err(_) => None,
        }
    }

    /// Sets the contents of the given index to the given item, verifying that
    /// the set will still be sorted and deduplicated afterwards. This is the
    /// preferred way to mutate items in the set.
    ///
    /// Returns an error if the new item doesn't compare greater than the item at
    /// index - 1 or if the item doesn't compare less than the item at index + 1.
    /// No data will be modified if an error is returned.
    pub fn set_item(&mut self, item: T, index: usize) -> Result<()> {
        if index > 0 {
            match T::compare(&item, &self.inner[index - 1]) {
                Ordering::Less => return Err(VerificationError::ItemTooSmall),
                Ordering::Equal => return Err(VerificationError::DuplicateData),
                Ordering::Greater => {}
            }
        }
        if index + 1 < self.len() {
            match T::compare(&item, &self.inner[index + 1]) {
                Ordering::Less => {}
                Ordering::Equal => return Err(VerificationError::DuplicateData),
                Ordering::Greater => return Err(VerificationError::ItemTooBig),
            }
        }
        self.inner[index] = item;
        Ok(())
    }

    /// Sets the contents of the given index to the current item.
    ///
    /// Callers must either ensure that the new item compares greater than
    /// the item at index - 1 and less than the item at index + 1, or call
    ///  [`Self::verify()`] after every call to this (or any other
    /// `..._unchecked`) method.
    pub fn set_item_unchecked(&mut self, item: T, index: usize) {
        self.inner[index] = item;
    }

    /// Replaces an item with a new item that compares as equal. This method
    /// is only useful if it is possible to have 2 conceptually different
    /// instances of T that compare equal.
    ///
    /// Returns an error if no items in the set compare equal to the given
    /// item.
    pub fn replace(&mut self, item: T) -> Result<()> {
        match self.binary_search_item(&item) {
            Ok(index) => {
                self.inner[index] = item;
                Ok(())
            }
            Err(_) => Err(VerificationError::NotFound),
        }
    }

    /// Removes a single item by key.
    ///
    /// Returns None if the key doesn't match any item in the set.
    pub fn remove_key(&mut self, key: &T::Key) -> Option<T> {
        match self.binary_search_key(key) {
            Ok(index) => Some(self.inner.remove(index)),
            Err(_) => None,
        }
    }

    /// Returns an iterator over the elements in both `OrdSetVec`s, specifying
    /// which of the input sets each item is included in. The return type of this
    /// method implements `Iterator<Item = Inclusion<T>>`.
    ///
    /// See [`iter_set::classify()`] and [`iter_set::classify_by()`] for more details.
    #[allow(clippy::type_complexity)]
    pub fn classify<'a>(
        &'a self,
        second: &'a OrdSetVec<T>,
    ) -> ClassifyBy<
        std::slice::Iter<'a, T>,
        std::slice::Iter<'a, T>,
        fn(&mut &T, &mut &T) -> Ordering,
    > {
        classify_by(self.inner.iter(), second.inner.iter(), |a, b| {
            T::compare(a, b)
        })
    }

    /// Clears the contents of this `OrdSetVec`.
    ///
    /// See [`Vec::clear()`].
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Call a closure on each element in this `OrdSetVec` mutably.
    ///
    /// After mutating an element it must compare greater than the previously encountered element,
    /// otherwise this function will panic.
    pub fn for_each_mut<F>(&mut self, mut f: F)
    where
        Self: Sized,
        F: FnMut(&mut T),
    {
        for x in 0..self.inner.len() {
            f(&mut self.inner[x]);
            if x > 0 {
                assert_eq!(
                    T::compare(&self.inner[x], &self.inner[x - 1]),
                    Ordering::Greater
                );
            }
        }
    }

    fn find_dup(slice: &[T]) -> Option<usize> {
        for i in 1..slice.len() {
            if T::compare(&slice[i - 1], &slice[i]) == Ordering::Equal {
                return Some(i);
            }
        }
        None
    }
}

/// Indicates that an iterator is ordered and de-duplicated.
/// Does not perform any checking by itself.
pub trait OrdSetIter<T: OrdSetItemTrait>: Iterator<Item = T> {
    /// Iterate over the union of this set and another that has no duplicates
    /// with this set.
    fn union_set<B>(self, other: B) -> UnionIter<T, Self, B>
    where
        B: OrdSetIter<T> + Iterator<Item = T>,
    {
        UnionIter::new(self, other)
    }

    /// Collect this set into an [`OrdSetVec`].
    fn collect_set(self) -> OrdSetVec<<Self as Iterator>::Item> {
        OrdSetVec {
            inner: Iterator::collect(self),
        }
    }

    /// Same as [`Iterator::map()`], but verifies that the resulting iterator is
    /// sorted and deduplicated.
    fn map_and_verify<F>(self, f: F) -> OrdSetIterVerify<T, std::iter::Map<Self, F>>
    where
        F: FnMut(Self::Item) -> T,
    {
    }
}

/// Iterator over the union of two [`OrdSetIter`]s that have no duplicates
/// between them
pub struct UnionIter<T, A, B>
where
    T: OrdSetItemTrait,
    A: OrdSetIter<T> + Iterator<Item = T>,
    B: OrdSetIter<T> + Iterator<Item = T>,
{
    a: Peekable<A>,
    b: Peekable<B>,
}

impl<T, A, B> UnionIter<T, A, B>
where
    T: OrdSetItemTrait,
    A: OrdSetIter<T> + Iterator<Item = T>,
    B: OrdSetIter<T> + Iterator<Item = T>,
{
    pub fn new(a: A, b: B) -> Self {
        Self {
            a: a.peekable(),
            b: b.peekable(),
        }
    }
}

impl<T, A, B> Iterator for UnionIter<T, A, B>
where
    T: OrdSetItemTrait,
    A: OrdSetIter<T> + Iterator<Item = T>,
    B: OrdSetIter<T> + Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let a = match self.a.peek() {
            None => return self.b.next(),
            Some(v) => v,
        };
        let b = match self.b.peek() {
            None => return self.a.next(),
            Some(v) => v,
        };
        match T::compare(a, b) {
            Ordering::Less => self.a.next(),
            Ordering::Greater => self.b.next(),
            Ordering::Equal => panic!("duplicate items in union"),
        }
    }
}

impl<T, A, B> OrdSetIter for UnionIter<T, A, B>
where
    T: OrdSetItemTrait,
    A: OrdSetIter<T> + Iterator<Item = T>,
    B: OrdSetIter<T> + Iterator<Item = T>,
{
}

pub struct OrdSetIterVerify<T, I>
where
    T: OrdSetItemTrait,
    I: Iterator<Item = T>,
{
    i: Peekable<I>,
}

impl<T, I> OrdSetIterVerify<T, I>
where
    T: OrdSetItemTrait,
    I: Iterator<Item = T>,
{
    pub fn new(i: I) -> Self {
        Self { i: i.peekable() }
    }
}

impl<T, I> Iterator for OrdSetIterVerify<T, I>
where
    T: OrdSetItemTrait,
    I: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let item = match self.i.next() {
            None => return None,
            Some(v) => v,
        };
        let peek = match self.i.peek() {
            None => return Some(item),
            Some(v) => v,
        };
        if T::compare(&item, &peek) != Ordering::Less {
            panic!("Unordered or duplicate data");
        }
        Some(item)
    }
}

impl<T, I> OrdSetIter for OrdSetIterVerify<T, I>
where
    T: OrdSetItemTrait,
    I: Iterator<Item = T>,
{
}
