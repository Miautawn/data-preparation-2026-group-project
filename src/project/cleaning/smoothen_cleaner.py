import numpy as np

class SmoothenCleaner:
    def __init__(self, max_diff: float):
        """
        Initialize the SmoothenCleaner with a max_diff parameter.
        
        Args:
            max_diff (float): The maximum allowed distance between two consecutive elements.
        """
        self.max_diff = max_diff

    def clean_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Smoothens an array by capping the distance between consecutive elements.
        If the distance between x_i and x_i+1 is greater than max_diff,
        it keeps the direction but reduces the distance to the previous distance 
        (if any) or to max_diff.
        
        Args:
            arr (np.ndarray): The input array to clean.
            
        Returns:
            np.ndarray: The smoothened array.
        """
        if len(arr) <= 1:
            return np.array(arr)

        cleaned = np.array(arr, dtype=float)
        
        # Traverse from the beginning
        for i in range(len(arr) - 1):
            e_i = arr[i]
            e_next = arr[i+1]
            c_i = cleaned[i]
            
            # Original jump
            diff = e_next - e_i
            dist = abs(diff)
            direction = np.sign(diff) if diff != 0 else 1.0

            if dist > self.max_diff:
                # Use previous cleaned distance if available, otherwise max_diff
                if i > 0:
                    target_dist = abs(cleaned[i] - cleaned[i-1])
                else:
                    target_dist = self.max_diff
                
                # Cap the target distance at max_diff if it was also a spike or if we want strict capping
                # However, the prompt says "reduce to previous distance OR max_diff (if previous not present)"
                # which implies we use the previous distance even if it was large? 
                # Usually "smoothen" means we want to avoid large jumps.
                # If the previous distance was also > max_diff, it would have been capped.
                # So cleaned[i] - cleaned[i-1] should already be "smooth".
                
                cleaned[i+1] = c_i + direction * target_dist
            else:
                # Keep the original jump but apply it to the cumulative cleaned value
                cleaned[i+1] = c_i + diff

        return cleaned
