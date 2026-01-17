import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import time
from collections import deque

class ASLDetector:
    """Simple and effective ASL detector"""
    
    def __init__(self, model_path='asl_model_final.h5', class_indices_path='class_indices.json'):
        """Initialize detector"""
        print("\n" + "="*60)
        print("🚀 Loading ASL Detector...")
        print("="*60)
        
        # Load model
        print("📦 Loading model...")
        self.model = keras.models.load_model(model_path)
        print("✅ Model loaded!")
        
        # Load class mappings
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        self.idx_to_class = {v: k for k, v in class_indices.items()}
        print(f"✅ {len(self.idx_to_class)} classes loaded")
        
        self.img_size = 224
        self.prediction_history = deque(maxlen=10)
        
        # Fixed ROI box settings (where you place your hand)
        self.roi_size = 300  # Size of the square ROI
        
        # Prediction smoothing - average last N predictions
        self.recent_predictions = deque(maxlen=5)  # Last 5 predictions
        self.prediction_buffer = {}  # Count predictions
        
        print("="*60 + "\n")
    
    def preprocess_for_model(self, roi):
        """Preprocess ROI exactly as training data was prepared"""
        # Resize to model input size
        img = cv2.resize(roi, (self.img_size, self.img_size))
        
        # Convert BGR to RGB (important!)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def get_smoothed_prediction(self, prediction, confidence):
        """Smooth predictions by averaging recent results"""
        # Add to recent predictions
        self.recent_predictions.append((prediction, confidence))
        
        # Count each prediction
        pred_counts = {}
        total_conf = {}
        
        for pred, conf in self.recent_predictions:
            if conf > 0.3:  # Only count if somewhat confident
                pred_counts[pred] = pred_counts.get(pred, 0) + 1
                total_conf[pred] = total_conf.get(pred, 0) + conf
        
        if not pred_counts:
            return prediction, confidence
        
        # Get most common prediction
        best_pred = max(pred_counts, key=pred_counts.get)
        avg_conf = total_conf[best_pred] / pred_counts[best_pred]
        
        # Only return if it appears in majority of recent frames
        if pred_counts[best_pred] >= 3:  # At least 3 out of 5
            return best_pred, avg_conf
        
        return prediction, confidence
    
    def predict(self, roi):
        """Make prediction on ROI"""
        processed = self.preprocess_for_model(roi)
        predictions = self.model.predict(processed, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        predicted_class = self.idx_to_class[predicted_idx]
        
        # Get top 3 predictions for debugging
        top3_idx = np.argsort(predictions[0])[-3:][::-1]
        top3 = [(self.idx_to_class[i], predictions[0][i]) for i in top3_idx]
        
        return predicted_class, confidence, top3
    
    def draw_roi_box(self, frame, x, y, size):
        """Draw the ROI box where user should place hand"""
        # Draw thick border
        cv2.rectangle(frame, (x, y), (x + size, y + size), (0, 255, 0), 3)
        
        # Draw corner markers
        corner_len = 30
        # Top-left
        cv2.line(frame, (x, y), (x + corner_len, y), (0, 255, 0), 5)
        cv2.line(frame, (x, y), (x, y + corner_len), (0, 255, 0), 5)
        # Top-right
        cv2.line(frame, (x + size, y), (x + size - corner_len, y), (0, 255, 0), 5)
        cv2.line(frame, (x + size, y), (x + size, y + corner_len), (0, 255, 0), 5)
        # Bottom-left
        cv2.line(frame, (x, y + size), (x + corner_len, y + size), (0, 255, 0), 5)
        cv2.line(frame, (x, y + size), (x, y + size - corner_len), (0, 255, 0), 5)
        # Bottom-right
        cv2.line(frame, (x + size, y + size), (x + size - corner_len, y + size), (0, 255, 0), 5)
        cv2.line(frame, (x + size, y + size), (x + size, y + size - corner_len), (0, 255, 0), 5)
        
        # Instruction text
        cv2.putText(frame, "Place hand here", (x + 10, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def draw_ui(self, frame, prediction, confidence, top3, fps):
        """Draw UI elements"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay for header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (40, 40, 40), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
        
        # Main prediction
        if confidence > 0.8:
            color = (0, 255, 0)  # Green - high confidence
            status = "✓ CONFIDENT"
        elif confidence > 0.5:
            color = (0, 200, 255)  # Orange - medium
            status = " MAYBE"
        else:
            color = (0, 0, 255)  # Red - low confidence
            status = " LOW CONF"
        
        # Large prediction text
        cv2.putText(frame, f"Sign: {prediction}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        cv2.putText(frame, f"{status} ({confidence:.1%})", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Confidence bar
        bar_y = 110
        bar_width = int(400 * confidence)
        cv2.rectangle(frame, (20, bar_y), (420, bar_y + 15), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, bar_y), (20 + bar_width, bar_y + 15), color, -1)
        
        # Top 3 predictions (for debugging)
        cv2.putText(frame, "Top 3:", (20, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        top3_text = " | ".join([f"{cls}:{conf:.0%}" for cls, conf in top3])
        cv2.putText(frame, top3_text, (80, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS (top right)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Bottom bar for history
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (40, 40, 40), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
        
        # History
        if len(self.prediction_history) > 0:
            history_text = "History: " + " → ".join(list(self.prediction_history))
        else:
            history_text = "History: (empty)"
        
        cv2.putText(frame, history_text, (10, h - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Controls
        cv2.putText(frame, "Q:Quit  C:Clear  S:Screenshot  SPACE:Pause", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        return frame
    
    def run(self):
        """Main detection loop"""
        # Open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot access camera!")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("🎥 Camera started!")
        print("\n📋 INSTRUCTIONS:")
        print("   1. Place your hand inside the GREEN BOX")
        print("   2. Make clear ASL signs")
        print("   3. Hold steady for 1-2 seconds")
        print("\n⌨️  CONTROLS:")
        print("   Q - Quit")
        print("   C - Clear history")
        print("   S - Save screenshot")
        print("   SPACE - Pause/Resume")
        print("\n" + "="*60 + "\n")
        
        # Variables
        prev_time = time.time()
        prev_prediction = None
        screenshot_count = 0
        paused = False
        frame_count = 0
        
        # Create window
        cv2.namedWindow('ASL Detector', cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Flip horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                
                # Calculate ROI position (center of frame)
                roi_x = (w - self.roi_size) // 2
                roi_y = (h - self.roi_size) // 2
                
                # Extract ROI
                roi = frame[roi_y:roi_y + self.roi_size, roi_x:roi_x + self.roi_size]
                
                # Make prediction if not paused
                if not paused and frame_count % 2 == 0:  # Predict every 2nd frame
                    raw_prediction, raw_confidence, top3 = self.predict(roi)
                    
                    # Apply smoothing
                    prediction, confidence = self.get_smoothed_prediction(raw_prediction, raw_confidence)
                    
                    # Add to history if confident and stable
                    if confidence > 0.80 and prediction != prev_prediction:
                        self.prediction_history.append(prediction)
                        prev_prediction = prediction
                else:
                    if 'prediction' not in locals():
                        prediction = "WAITING"
                        confidence = 0.0
                        top3 = [("", 0), ("", 0), ("", 0)]
                
                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                frame_count += 1
                
                # Draw ROI box
                self.draw_roi_box(frame, roi_x, roi_y, self.roi_size)
                
                # Draw UI
                frame = self.draw_ui(frame, prediction, confidence, top3, fps)
                
                # Show pause overlay
                if paused:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
                    cv2.putText(frame, "PAUSED", (w//2 - 150, h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                    cv2.putText(frame, "Press SPACE to resume", (w//2 - 200, h//2 + 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Display
                cv2.imshow('ASL Detector', frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n👋 Quitting...")
                    break
                elif key == ord('c') or key == ord('C'):
                    self.prediction_history.clear()
                    prev_prediction = None
                    print("🗑️  History cleared")
                elif key == ord('s') or key == ord('S'):
                    screenshot_count += 1
                    filename = f"asl_screenshot_{screenshot_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == 32:  # SPACE
                    paused = not paused
                    print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "="*60)
            print("✅ Session ended")
            if len(self.prediction_history) > 0:
                print(f"📊 Detected signs: {' → '.join(list(self.prediction_history))}")
            print("="*60 + "\n")

def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("      ASL ALPHABET DETECTION")
    print("="*60)
    
    try:
        detector = ASLDetector()
        detector.run()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n📁 Required files:")
        print("   - best_asl_model.h5")
        print("   - class_indices.json")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()