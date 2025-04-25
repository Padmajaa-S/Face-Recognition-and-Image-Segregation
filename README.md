# ReFace - Face Recognition and Image Segregation

This project uses Dlib's pre-trained face recognition models to detect and recognize faces in a new photo album. The images are then segregated based on the people present in them by creating separate folders for each recognized person.

## Features

- Detects faces in images using Dlib's pre-trained face detector.
- Recognizes faces using Dlib's ResNet-based face recognition model.
- Segregates images based on recognized faces by creating separate folders for each person.
- Organizes unrecognized faces into an "Unknown" folder.

## Project Structure

- `face_recognition.py`: The main Python script that runs the face recognition and image segregation.
- `known_faces/`: Directory containing known faces with subdirectories for each person.
- `new_album/`: Directory containing the new photo album to be processed.
- `organized_faces/`: Output directory where images are segregated based on recognized faces.

## Usage

1. Prepare the `known_faces` directory:
    - Create subdirectories for each person and place their images inside.
    - Example structure:
      ```
      known_faces/
      ├── Alice/
      │   ├── alice1.jpg
      │   ├── alice2.jpg
      ├── Bob/
      │   ├── bob1.jpg
      │   ├── bob2.jpg
      ```

2. Prepare the `new_album` directory:
    - Place all the images from the new photo album inside this directory.

3. Update the paths in the script (`face_recognition.py`):
    ```python
    predictor_path = r"C:\path\to\shape_predictor_68_face_landmarks.dat"
    face_rec_model_path = r"C:\path\to\dlib_face_recognition_resnet_model_v1.dat"
    known_faces_dir = r"C:\path\to\known_faces"
    new_album_dir = r"D:\path\to\new_album"
    output_base_dir = r'C:\path\to\organized_faces'
    ```

4. Run the face recognition and image segregation script:
    ```sh
    python face_recognition.py
    ```

5. The script will process the images in the `new_album` directory, recognize faces, and organize the images in the `organized_faces` directory.

## Code Overview

### Main Steps

1. **Load Pre-trained Models**:
    - Load Dlib's face detector, shape predictor, and face recognition model.

    ```python
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
    ```

2. **Extract Known Face Descriptors**:
    - Extract face descriptors for known faces and store them along with the corresponding names.

    ```python
    known_face_descriptors = []
    known_face_names = []

    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                filepath = os.path.join(person_dir, filename)
                image = cv2.imread(filepath)
                faces = detector(image, 1)
                if len(faces) == 1:
                    descriptor = get_face_descriptor(image, faces[0])
                    known_face_descriptors.append(descriptor)
                    known_face_names.append(person_name)
    ```

3. **Detect and Recognize Faces in New Album**:
    - Detect faces in each image from the new album and recognize them by comparing with known face descriptors.
    - Organize images based on recognized faces.

    ```python
    def detect_and_recognize_faces(image_path, threshold=0.6):
        image = cv2.imread(image_path)
        faces = detector(image, 1)
        face_labels = []
        for face in faces:
            descriptor = get_face_descriptor(image, face)
            distances = np.linalg.norm(known_face_descriptors - descriptor, axis=1)
            min_distance_index = np.argmin(distances)
            confidence = 1 - distances[min_distance_index]
            if distances[min_distance_index] < threshold:
                name = known_face_names[min_distance_index]
            else:
                name = "Unknown"
            face_labels.append(name)
        return face_labels

    def process_album(input_dir, output_base_dir, album_name, threshold=0.6):
        output_dir = os.path.join(output_base_dir, album_name)
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_dir, filename)
                face_labels = detect_and_recognize_faces(image_path, threshold)
                unique_labels = set(face_labels)
                for label in unique_labels:
                    label_dir = os.path.join(output_dir, label)
                    if not os.path.exists(label_dir):
                        os.makedirs(label_dir)
                    shutil.copy(image_path, label_dir)
    ```

### Execution

1. Call the `process_album` function to start the face recognition and image segregation process.

    ```python
    album_name = 'album_matches'
    process_album(new_album_dir, output_base_dir, album_name)
    ```

## License

This project is licensed under the MIT License.

## Contact

For any questions or inquiries, please contact [padmajaatms@gmail.com](padmajaatms@gmail.com).
