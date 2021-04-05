import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3-custom.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/y2.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 1, 'number of classes in the model')


def main(_argv):
    isFirst = True # 비디오 파일 첫 화면

    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0

    #initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                physical_devices[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        except RuntimeError as e:
            print(e)

    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #if len(physical_devices) > 0:
    #    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0
    count = 0 
    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else:
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # SELECT ROI FOR EXCEPT FACE
        key = cv2.waitKey(1) & 0xff
        if key == ord(' ') or (FLAGS.video != 0 and isFirst):
            isFirst = False
            except_img = np.zeros((height, width, 3), np.uint8)
            #cv2.imshow('test', except_img)
            roi = cv2.selectROI('output', img, False)
            crop_img = img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

        if crop_img.size > 0:
            except_img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = crop_img
            #cv2.imshow('test', except_img)
            crop_img_in = cv2.cvtColor(except_img, cv2.COLOR_BGR2RGB)
            crop_img_in = tf.expand_dims(crop_img_in, 0)
            crop_img_in = transform_images(crop_img_in, FLAGS.size)

            c_boxes, c_scores, c_classes, c_nums = yolo.predict(crop_img_in)
            c_classes = c_classes[0]
            c_names = []
            for i in range(len(c_classes)):
                c_names.append(class_names[int(c_classes[i])])
            c_converted_boxes = convert_boxes(except_img, c_boxes[0])
            c_features = encoder(except_img, c_converted_boxes)
            c_detections = [Detection(bbox, score, class_name, feature) for
                            bbox, score, class_name, feature in
                            zip(c_converted_boxes, c_scores[0], c_names, c_features)]


        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        #Mosaic others
        if c_detections:
            for c_det in c_detections:
                for det in detections:
                    cos_sil = sum(det.feature * c_det.feature)
                    if cos_sil < 0.8:  # 유사도가 0.9 보다 낮으면 모자이크 처리
                        blur_image = img[int(det.tlwh[1]):int(det.tlwh[1] + det.tlwh[3]),
                                    int(det.tlwh[0]):int(det.tlwh[0] + det.tlwh[2])]
                        blur_image = cv2.resize(blur_image, dsize=(0, 0), fx=0.04, fy=0.04)
                        blur_image = cv2.resize(blur_image, (int(det.tlwh[2]), int(det.tlwh[3])),
                                               interpolation=cv2.INTER_AREA)
                        img[int(det.tlwh[1]):int(det.tlwh[1] + det.tlwh[3]),
                        int(det.tlwh[0]):int(det.tlwh[0] + det.tlwh[2])] = blur_image

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            #cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            #if(track.track_id == c_track_id):
            #    blur_image = img[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
            #    blur_image = cv2.blur(blur_image, (60, 60))
            #    img[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])] = blur_image


        ### UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
        #for det in detections:
        #    bbox = det.to_tlbr()
        #    cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

        # print fps on screen 
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('output', img)

        if FLAGS.output:
            out.write(img)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(converted_boxes) != 0:
                for i in range(0,len(converted_boxes)):
                    list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
            list_file.write('\n')

        # press q to quit
        if key == ord('q'):
            break

    vid.release()
    if FLAGS.output:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
