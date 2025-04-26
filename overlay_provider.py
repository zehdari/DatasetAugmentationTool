import os
import threading
import queue
import random
import cv2

class OverlayProvider:
    """
    Keeps up to `max_buffers` overlay images pre-loaded in a thread-safe queue.
    Images are cycled without repeats until the entire pool has been used.
    """

    def __init__(self, image_folder: str, max_buffers: int = 10):
        self.image_folder = image_folder
        # list of all files in the folder
        self.files = [f for f in os.listdir(image_folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                      and os.path.isfile(os.path.join(image_folder, f))]
        if not self.files:
            raise RuntimeError(f"No images found in {image_folder}")

        # a shuffled pool of indices, so we never repeat until exhausted
        self._lock = threading.Lock()
        self.idx_pool = list(range(len(self.files)))
        random.shuffle(self.idx_pool)

        # a bounded queue that holds CV2 images
        self.q = queue.Queue(maxsize=max_buffers)
        self._shutdown = threading.Event()

        # start the background loader
        self.loader = threading.Thread(target=self._loader, daemon=True)
        self.loader.start()

    def _loader(self):
        while not self._shutdown.is_set():
            try:
                # block until there’s space in the queue
                img = self._load_one()
                self.q.put(img, timeout=0.1)
            except queue.Full:
                # queue is full, retry until shutdown
                continue

    def _load_one(self):
        # refill & reshuffle when we’ve used all
        with self._lock:
            if not self.idx_pool:
                self.idx_pool = list(range(len(self.files)))
                random.shuffle(self.idx_pool)
            idx = self.idx_pool.pop()

        path = os.path.join(self.image_folder, self.files[idx])
        img = cv2.imread(path)
        if img is None:
            # if a file failed to read, skip it
            return self._load_one()
        return img

    def get_overlay(self, timeout=None):
        """
        Retrieve the next pre-loaded image (blocks until one’s available).
        Returns None only if `timeout` is given and expires.
        """
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        """Shut down the loader thread and drain the queue."""
        self._shutdown.set()
        self.loader.join()
        # clear out any images
        while not self.q.empty():
            self.q.get_nowait()