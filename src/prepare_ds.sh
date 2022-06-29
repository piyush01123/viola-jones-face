
mkdir -p data/face_detection_dataset/train/face
mkdir -p data/face_detection_dataset/train/nonface
mkdir -p data/face_detection_dataset/test/face
mkdir -p data/face_detection_dataset/test/nonface

cp data/SUN_data/train/*/* data/face_detection_dataset/train/nonface/
cp data/SUN_data/test/*/* data/face_detection_dataset/test/nonface/
ls data/selected_faces/*.jpg | tail -n 150 | xargs -I {} bash -c "cp {} data/face_detection_dataset/test/face/"
cp data/selected_faces/*.jpg data/face_detection_dataset/train/face/
