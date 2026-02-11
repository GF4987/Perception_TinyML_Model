"""
TinyML Face Detection - Laptop Version
Run MobileNetV2-SSD face detection directly on your laptop with webcam
"""

import tensorflow as tf
import numpy as np
import cv2
import time
import os

print(f"TensorFlow version: {tf.__version__}")
print(f"OpenCV version: {cv2.__version__}")

# Configuration
class Config:
    INPUT_SIZE = 96
    NUM_CLASSES = 2  # background + face
    ALPHA = 0.35
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4
    CAMERA_INDEX = 0  # Default webcam
    DISPLAY_SCALE = 4  # Scale up the small 96x96 image for display

config = Config()


def _make_divisible(v, divisor, min_value=None):
    """Ensure layers have channels divisible by divisor"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def inverted_residual_block(x, expanded_channels, output_channels, stride, block_id):
    """Inverted residual block for MobileNetV2"""
    in_channels = x.shape[-1]
    
    if block_id:
        x_expand = tf.keras.layers.Conv2D(expanded_channels, 1, padding='same', 
                                          use_bias=False, name=f'block_{block_id}_expand')(x)
        x_expand = tf.keras.layers.BatchNormalization(name=f'block_{block_id}_expand_BN')(x_expand)
        x_expand = tf.keras.layers.ReLU(6., name=f'block_{block_id}_expand_relu')(x_expand)
    else:
        x_expand = x
    
    x_depthwise = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding='same',
                                                   use_bias=False, 
                                                   name=f'block_{block_id}_depthwise')(x_expand)
    x_depthwise = tf.keras.layers.BatchNormalization(name=f'block_{block_id}_depthwise_BN')(x_depthwise)
    x_depthwise = tf.keras.layers.ReLU(6., name=f'block_{block_id}_depthwise_relu')(x_depthwise)
    
    x_project = tf.keras.layers.Conv2D(output_channels, 1, padding='same', use_bias=False,
                                       name=f'block_{block_id}_project')(x_depthwise)
    x_project = tf.keras.layers.BatchNormalization(name=f'block_{block_id}_project_BN')(x_project)
    
    if stride == 1 and in_channels == output_channels:
        return tf.keras.layers.Add(name=f'block_{block_id}_add')([x, x_project])
    else:
        return x_project


def create_mobilenetv2_backbone(input_shape, alpha=0.35):
    """Create MobileNetV2 backbone"""
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = tf.keras.layers.Conv2D(first_block_filters, 3, strides=2, padding='same',
                               use_bias=False, name='Conv1')(inputs)
    x = tf.keras.layers.BatchNormalization(name='bn_Conv1')(x)
    x = tf.keras.layers.ReLU(6., name='Conv1_relu')(x)
    
    inverted_residual_setting = [
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]
    
    feature_maps = []
    block_id = 0
    
    for t, c, n, s in inverted_residual_setting:
        output_channel = _make_divisible(c * alpha, 8)
        
        for i in range(n):
            stride = s if i == 0 else 1
            expanded_channels = int(x.shape[-1] * t)
            x = inverted_residual_block(x, expanded_channels, output_channel, 
                                       stride, block_id)
            block_id += 1
        
        if block_id in [3, 6, 13, 17]:
            feature_maps.append(x)
    
    x = tf.keras.layers.Conv2D(_make_divisible(512 * alpha, 8), 1, padding='same',
                               name='Conv_extra_1')(x)
    x = tf.keras.layers.ReLU(6., name='Conv_extra_1_relu')(x)
    feature_maps.append(x)
    
    x = tf.keras.layers.Conv2D(_make_divisible(256 * alpha, 8), 1, padding='same',
                               name='Conv_extra_2')(x)
    x = tf.keras.layers.ReLU(6., name='Conv_extra_2_relu')(x)
    feature_maps.append(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=feature_maps, name='MobileNetV2_Backbone')
    return model


def create_prediction_head(feature_map, num_boxes, num_classes, name_prefix):
    """Create classification and localization prediction heads"""
    conf = tf.keras.layers.Conv2D(num_boxes * num_classes, 3, padding='same',
                                  name=f'{name_prefix}_conf')(feature_map)
    
    loc = tf.keras.layers.Conv2D(num_boxes * 4, 3, padding='same',
                                 name=f'{name_prefix}_loc')(feature_map)
    
    return conf, loc


def create_face_detector(input_shape=(96, 96, 3), num_classes=2, alpha=0.35):
    """Create complete face detection model"""
    num_boxes = [3, 6, 6, 6, 6, 6]
    
    backbone = create_mobilenetv2_backbone(input_shape, alpha)
    feature_maps = backbone.output
    
    all_conf_outputs = []
    all_loc_outputs = []
    
    for i, feature_map in enumerate(feature_maps):
        conf, loc = create_prediction_head(feature_map, num_boxes[i], num_classes, 
                                          f'feature_map_{i}')
        
        conf_reshaped = tf.keras.layers.Reshape((-1, num_classes), 
                                               name=f'conf_reshape_{i}')(conf)
        loc_reshaped = tf.keras.layers.Reshape((-1, 4), 
                                              name=f'loc_reshape_{i}')(loc)
        
        all_conf_outputs.append(conf_reshaped)
        all_loc_outputs.append(loc_reshaped)
    
    conf_output = tf.keras.layers.Concatenate(axis=1, name='conf_concat')(all_conf_outputs)
    loc_output = tf.keras.layers.Concatenate(axis=1, name='loc_concat')(all_loc_outputs)
    
    conf_output = tf.keras.layers.Activation('softmax', name='conf_softmax')(conf_output)
    
    model = tf.keras.Model(inputs=backbone.input, 
                          outputs=[conf_output, loc_output],
                          name='FaceDetector')
    
    return model


def generate_default_boxes(num_boxes_total, img_size=96):
    """Generate default anchor boxes for SSD - simplified version"""
    # Create a grid of default boxes evenly distributed
    # This is a simplified approach that matches any output size
    default_boxes = []
    
    # Calculate grid size
    grid_size = int(np.sqrt(num_boxes_total / 3))  # Approximate grid
    if grid_size < 1:
        grid_size = 1
    
    # Generate boxes in a grid pattern
    for i in range(grid_size):
        for j in range(grid_size):
            cx = (j + 0.5) / grid_size
            cy = (i + 0.5) / grid_size
            
            # Multiple scales and aspect ratios
            for scale in [0.1, 0.3, 0.5]:
                for ratio in [1.0, 1.5, 2.0]:
                    if len(default_boxes) >= num_boxes_total:
                        break
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)
                    default_boxes.append([cx, cy, w, h])
                if len(default_boxes) >= num_boxes_total:
                    break
            if len(default_boxes) >= num_boxes_total:
                break
        if len(default_boxes) >= num_boxes_total:
            break
    
    # Pad if needed
    while len(default_boxes) < num_boxes_total:
        default_boxes.append([0.5, 0.5, 0.1, 0.1])
    
    return np.array(default_boxes[:num_boxes_total], dtype=np.float32)


def decode_predictions(conf_pred, loc_pred, default_boxes, 
                      conf_threshold=0.5, iou_threshold=0.4):
    """Decode model predictions to bounding boxes"""
    # Get face class confidences (index 1)
    face_scores = conf_pred[:, 1]
    
    # Filter by confidence threshold
    mask = face_scores > conf_threshold
    filtered_scores = face_scores[mask]
    filtered_boxes = loc_pred[mask]
    filtered_default_boxes = default_boxes[mask]
    
    if len(filtered_scores) == 0:
        return [], []
    
    # Decode bounding boxes
    decoded_boxes = []
    for i in range(len(filtered_scores)):
        # Convert from offset to absolute coordinates
        cx = filtered_boxes[i, 0] * 0.1 * filtered_default_boxes[i, 2] + filtered_default_boxes[i, 0]
        cy = filtered_boxes[i, 1] * 0.1 * filtered_default_boxes[i, 3] + filtered_default_boxes[i, 1]
        w = np.exp(filtered_boxes[i, 2] * 0.2) * filtered_default_boxes[i, 2]
        h = np.exp(filtered_boxes[i, 3] * 0.2) * filtered_default_boxes[i, 3]
        
        # Convert to corner format
        x1 = max(0, cx - w/2)
        y1 = max(0, cy - h/2)
        x2 = min(1, cx + w/2)
        y2 = min(1, cy + h/2)
        
        decoded_boxes.append([x1, y1, x2, y2])
    
    decoded_boxes = np.array(decoded_boxes)
    
    # Apply NMS
    indices = non_max_suppression(decoded_boxes, filtered_scores, iou_threshold)
    
    final_boxes = decoded_boxes[indices]
    final_scores = filtered_scores[indices]
    
    return final_boxes, final_scores


def non_max_suppression(boxes, scores, iou_threshold):
    """Non-maximum suppression to remove overlapping boxes"""
    if len(boxes) == 0:
        return []
    
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        ious = compute_iou(current_box, other_boxes)
        
        indices = indices[1:][ious < iou_threshold]
    
    return keep


def compute_iou(box, boxes):
    """Compute IoU between one box and multiple boxes"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    union = box_area + boxes_area - intersection
    
    iou = intersection / (union + 1e-6)
    
    return iou


class FaceDetectorDemo:
    """Face detector with webcam support"""
    
    def __init__(self, model_path=None):
        print("\n[1/3] Initializing face detector...")
        
        # Create or load model
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            self.model = tf.keras.models.load_model(model_path)
        else:
            print("Creating new model (untrained - for architecture demo)...")
            self.model = create_face_detector(
                input_shape=(config.INPUT_SIZE, config.INPUT_SIZE, 3),
                num_classes=config.NUM_CLASSES,
                alpha=config.ALPHA
            )
        
        print(f"Model parameters: {self.model.count_params():,}")
        
        # Get actual number of boxes from model output
        # Run a dummy prediction to get output shape
        dummy_input = np.zeros((1, config.INPUT_SIZE, config.INPUT_SIZE, 3), dtype=np.float32)
        conf_out, loc_out = self.model.predict(dummy_input, verbose=0)
        num_boxes = conf_out.shape[1]
        
        print(f"Model output boxes: {num_boxes}")
        
        # Generate default boxes matching the model output
        self.default_boxes = generate_default_boxes(num_boxes, config.INPUT_SIZE)
        
        print(f"Default boxes generated: {len(self.default_boxes)}")
        
        # Initialize camera
        print("\n[2/3] Initializing camera...")
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera!")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Camera initialized successfully")
        print("\n[3/3] Setup complete!\n")
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Resize to model input size
        resized = cv2.resize(frame, (config.INPUT_SIZE, config.INPUT_SIZE))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        # Preprocess
        input_data = self.preprocess_frame(frame)
        
        # Inference
        conf_pred, loc_pred = self.model.predict(input_data, verbose=0)
        
        # Decode predictions
        boxes, scores = decode_predictions(
            conf_pred[0], 
            loc_pred[0], 
            self.default_boxes,
            config.CONFIDENCE_THRESHOLD,
            config.IOU_THRESHOLD
        )
        
        return boxes, scores
    
    def draw_detections(self, frame, boxes, scores):
        """Draw bounding boxes on frame"""
        h, w = frame.shape[:2]
        
        for box, score in zip(boxes, scores):
            # Convert normalized coordinates to pixel coordinates
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence score
            label = f"Face: {score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # Get text size for background
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Draw background rectangle
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), font_thickness)
        
        return frame
    
    def draw_info(self, frame, num_faces):
        """Draw info overlay"""
        # FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        # Face count
        count_text = f"Faces: {num_faces}"
        cv2.putText(frame, count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        # Instructions
        instructions = "Press 'q' to quit | 's' to save frame"
        cv2.putText(frame, instructions, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed
    
    def run(self):
        """Run face detection on webcam"""
        print("=" * 60)
        print("FACE DETECTION DEMO")
        print("=" * 60)
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("\nNote: Model is untrained, so detections are random.")
        print("Train the model first for real face detection!")
        print("=" * 60)
        print("\nStarting detection...\n")
        
        saved_frame_count = 0
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Detect faces
                boxes, scores = self.detect_faces(frame)
                
                # Draw detections
                frame = self.draw_detections(frame, boxes, scores)
                
                # Draw info
                frame = self.draw_info(frame, len(boxes))
                
                # Update FPS
                self.update_fps()
                
                # Display
                cv2.imshow('TinyML Face Detection', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    filename = f"detected_face_{saved_frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame: {filename}")
                    saved_frame_count += 1
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            print("\nCleaning up...")
            self.cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "=" * 60)
            print(f"Session Summary:")
            print(f"  Frames processed: {self.frame_count}")
            print(f"  Average FPS: {self.fps:.1f}")
            print(f"  Frames saved: {saved_frame_count}")
            print("=" * 60)


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("TinyML Face Detection - Laptop Demo")
    print("=" * 60)
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nGPU available: {len(gpus)} device(s)")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("\nNo GPU found, using CPU")
    
    # Create and run detector
    try:
        detector = FaceDetectorDemo()
        detector.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()