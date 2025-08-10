import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict


class SimpleTrackableObject:
    """A simple class to store the centroid of a tracked object."""
    
    def __init__(self, object_id, centroid):
        self.object_id = object_id
        self.centroids = [centroid]
        self.counted = False


class SimpleCentroidTracker:
    """A simple centroid-based object tracker."""
    
    def __init__(self, max_disappeared=40, max_distance=75):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        """Register a new object with the tracker."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Deregister an object from the tracker."""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rectangles):
        """Update the tracker with new detections."""
        # If no objects are being tracked, register all new detections
        if len(self.objects) == 0:
            for rect in rectangles:
                centroid = self._get_centroid(rect)
                self.register(centroid)
        else:
            # Get centroids of current tracked objects
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Get centroids of new detections
            input_centroids = [self._get_centroid(rect) for rect in rectangles]
            
            # Compute distances between each pair of existing centroids and input centroids
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(input_centroids)):
                    D[i, j] = self._euclidean_distance(object_centroids[i], input_centroids[j])
            
            # Use Hungarian algorithm to find optimal assignment
            row_indices, col_indices = linear_sum_assignment(D)
            
            # Track which existing objects and input centroids have been used
            used_rows = set()
            used_cols = set()
            
            # Update existing objects
            for (row, col) in zip(row_indices, col_indices):
                if D[row, col] < self.max_distance:
                    object_id = object_ids[row]
                    self.objects[object_id] = input_centroids[col]
                    self.disappeared[object_id] = 0
                    used_rows.add(row)
                    used_cols.add(col)
            
            # Handle unused existing objects (mark as disappeared)
            unused_rows = set(range(len(object_centroids))) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new objects
            unused_cols = set(range(len(input_centroids))) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col])
        
        return self.objects

    def _get_centroid(self, rectangle):
        """Extract centroid from rectangle coordinates (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = rectangle
        return (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

    def _euclidean_distance(self, a, b):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
