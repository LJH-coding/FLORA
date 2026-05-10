# PyTorch
import torch
import imageio
import os
import datetime
import numpy as np
import glob

class RecordEnv:
    """Environment wrapper that records frames and saves them as GIF files."""

    def __init__(self, env, record_path=None, fps=60, max_gifs=5):
        self.env = env
        self.record_path = record_path
        self.fps = fps
        self.max_gifs = max_gifs
        self.frames = []
        self.total_reward = 0
        
        if self.record_path:
            os.makedirs(self.record_path, exist_ok=True)

    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def _cleanup_old_gifs(self):
        """Remove old GIF files to maintain FIFO with max_gifs limit"""
        if not self.record_path:
            return
            
        # Get all GIF files in the directory
        gif_pattern = os.path.join(self.record_path, "episode_*.gif")
        gif_files = glob.glob(gif_pattern)
        
        # Sort by creation time (oldest first)
        gif_files.sort(key=os.path.getctime)
        
        # Remove oldest files if we exceed the limit
        while len(gif_files) >= self.max_gifs:
            oldest_file = gif_files.pop(0)
            try:
                os.remove(oldest_file)
                print(f"Removed old GIF: {os.path.basename(oldest_file)}")
            except OSError as e:
                print(f"Failed to remove {oldest_file}: {e}")
    
    def step(self, action):
        obs = self.env.step(action)
        self.total_reward += obs.reward
        
        if self.record_path:
            # Get rendered frame
#            frame = obs.state
            frame = self.env.render()
#            self.frames.append(np.transpose(np.array(frame), (1, 2, 0)))
            self.frames.append(np.array(frame))
            
            # Save GIF when episode ends
            if obs.is_last and self.frames:
                # Clean up old GIFs before saving new one
                self._cleanup_old_gifs()
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"episode_{timestamp}_{self.total_reward}.gif"
                filepath = os.path.join(self.record_path, filename)
                imageio.mimsave(filepath, self.frames, fps=self.fps)
                print(f"Saved GIF: {filename}")
                self.frames = []
        
        return obs
    
    def reset(self):
        obs = self.env.reset()
        
        if self.record_path:
            self.frames = []
            self.total_reward = 0
#            frame = obs.state
            frame = self.env.render()
#            self.frames.append(np.transpose(np.array(frame), (1, 2, 0)))
            self.frames.append(np.array(frame))
        
        return obs