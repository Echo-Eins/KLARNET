// crates/buffering/src/ring_buffer.rs

pub struct RingBuffer<T> {
    buffer: Vec<T>,
    capacity: usize,
    write_pos: usize,
    read_pos: usize,
    size: usize,
}

impl<T: Clone + Default> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![T::default(); capacity],
            capacity,
            write_pos: 0,
            read_pos: 0,
            size: 0,
        }
    }

    pub fn push(&mut self, item: T) {
        self.buffer[self.write_pos] = item;
        self.write_pos = (self.write_pos + 1) % self.capacity;

        if self.size < self.capacity {
            self.size += 1;
        } else {
            self.read_pos = (self.read_pos + 1) % self.capacity;
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.size == 0 {
            return None;
        }

        let item = self.buffer[self.read_pos].clone();
        self.read_pos = (self.read_pos + 1) % self.capacity;
        self.size -= 1;

        Some(item)
    }

    pub fn get_last_n(&self, n: usize) -> Vec<T> {
        let count = n.min(self.size);
        let mut result = Vec::with_capacity(count);

        let start_pos = if self.write_pos >= count {
            self.write_pos - count
        } else {
            self.capacity - (count - self.write_pos)
        };

        for i in 0..count {
            let pos = (start_pos + i) % self.capacity;
            result.push(self.buffer[pos].clone());
        }

        result
    }

    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.read_pos = 0;
        self.size = 0;
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}