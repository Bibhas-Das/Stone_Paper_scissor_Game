import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Finger landmarks
finger_tip_ids = [4, 8, 12, 16, 20]  # Index finger to pinky finger tips
finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

def detect_fingers(landmarks):
    fingers = []
    # Thumb
    if landmarks[4].y < landmarks[3].y and landmarks[4].x > landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other fingers
    for i in range(1, 5):
        if landmarks[finger_tip_ids[i]].y < landmarks[finger_tip_ids[i]-2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands with custom parameters
with mp_hands.Hands(
    max_num_hands=1,                      # Maximum number of hands to detect
    min_detection_confidence=0.8,         # Minimum confidence for detection
    min_tracking_confidence=0.8) as hands: # Minimum confidence for tracking

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hands
        results = hands.process(image_rgb)

        # Draw hand landmarks on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Detect fingers
                finger_states = detect_fingers(hand_landmarks.landmark)
                finger_up_count = sum(finger_states)
                
                # Display finger states
                '''
                cv2.putText(frame, f'Fingers Up: {finger_up_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                for i, finger_state in enumerate(finger_states):
                    cv2.putText(frame, f'{finger_names[i]}: {finger_state}', (10, 30*(i+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                '''
                if finger_states in [[0,0,1,0,0],[1,0,1,0,0]]:
                    print("fuck")
                elif finger_states in [[0,0,1,1,1],[1,0,1,1,1]]:
                    print("nice")
                    #exit()
                elif finger_states in [[0,1,1,0,0],[1,1,1,0,0]]:
                    print("You : siger\nMe : Rock")
                elif finger_states in [[0,0,0,0,0],[1,0,0,0,0]]:
                    print("You : stone\nMe : Paper")
                elif finger_states in [[0,1,0,0,0],[1,1,0,0,0]]:
                    print("You : out")
                elif finger_states in [[1,1,1,1,1],[1,1,1,1,1]]:
                    print("You : paper\nMe : Siger")
                else:
                    print(finger_states)
                
        # Display the annotated image
        cv2.imshow('MediaPipe Hands', frame)

        # Press 'q' to exit the loop
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        cv2.waitKey(1)
# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
