import os
import imutils
import ffmpeg
import dlib
import random
import cv2

class VideoProcessor():
    def __init__(self, video_file, user_id=None, mark=None, output_folder=None, output_frames=6, shape_predictor='shape_predictor_68_face_landmarks.dat'):
        # by default video_file name is 'user_id.mp4'

        if user_id==None:
            self.user_id=os.path.splittext(os.path.basename(video_file))
        self.mark = mark
        self.output_frames = output_frames
        self.frames_to_process = int(output_frames*3.33)

        if output_folder==None:
            self.output_folder = os.path.curdir

        self.shape_predictor_path = shape_predictor
        if not os.path.exists(self.shape_predictor_path):
            cmd = 'wget -c --progress=bar http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && bzip2 -d shape_predictor_68_face_landmarks.dat.bz2'
            os.system(cmd)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor_path)

        cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
        haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(haar_model)

        if not os.path.exists(video_file):
            raise FileNotFoundError
        else:
            self.video_file = video_file
        self.angle = self.check_rotation()

    def check_rotation(self):
        meta_dict = ffmpeg.probe(self.video_file)
        angle = None
        if 'rotate' in meta_dict['streams'][0]['tags'].keys():
            angle = int(meta_dict['streams'][0]['tags']['rotate'])
        return angle

    def rotate_frame(self, image):
        return imutils.rotate_bound(image, -self.angle)

    def detect_by_dlib(self, image, frame):
        image = imutils.resize(image, width=240)
        rects = self.detector(image, 1)
        if len(rects) < 1:
            return None
        else:
            cv2.imwrite(os.path.join(self.output_folder, f'{self.user_id}_{frame}_{self.mark}.jpg'), image)
        return 1

    def soft_detect(self, image, frame):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 3)
        widths = [v[2] for v in [f for f in faces] if v[2]>150]
        if len(widths) < 1:
            return None
        else:
            cv2.imwrite(os.path.join(self.output_folder, f'{self.user_id}_{frame}_{self.mark}.jpg'), image)
        return 1

    def process_video_file(self):
        vidcap = cv2.VideoCapture(self.video_file)
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        indexes = [random.randint(0, length - 1) for _ in range(self.frames_to_process)]

        success, image = vidcap.read()
        count = 0
        out = 0
        while success:
            if count in indexes:
                image = self.rotate_frame(image)
                frame = indexes.index(count)
                res = self.detect_by_dlib(image, frame)
                if res is None:
                    res = self.soft_detect(image, frame)
                if res != None:
                    out += 1
            if out == self.output_frames:
                return 'Video successfully processed'
            success, image = vidcap.read()
            count += 1
        return f'Video failed. {out} frames with faces detected'


if __name__=="__main__":
    VideoProcessor('sdfs')