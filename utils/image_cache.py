import cv2

class ImageCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        
    def get_image(self, path):
        if path in self.cache:
            return self.cache[path].copy()
        
        image = cv2.imread(path)
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[path] = image
        return image.copy()

    def clear(self):
        self.cache.clear()