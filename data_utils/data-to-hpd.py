import argparse
import multiprocessing
import os
import sys
import time
import random
import cv2
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as et


def error(error_id):
    error_prefix = "[Error] "
    if error_id == -1:
        print(error_prefix, "Input Cvat dataset is not valid !")
    elif error_id == -2:
        print(error_prefix, "Output path already exist.\nPlease delete it first !")
    elif error_id == -3:
        print(error_prefix, "Input 2 Class KITTI dataset is not valid !")
    elif error_id == -4:
        print(error_prefix, "Invalid Augmentation type !")
    elif error_id == -5:
        print(error_prefix, "Invalid Mode !")
    elif error_id == -6:
        print(error_prefix, "Total sum of percentage Should be 100 !")
    elif error_id == -7:
        print(error_prefix, "Dataset Percentage should not be more than 100 in any augmentation !")
    else:
        print(error_prefix, "Unexpected Error encountered !")
    sys.exit()


def create_dir(path_list):
    for path in path_list:
        print("Creating Directory: {}".format(path))
        if os.path.exists(path):
            error(-2)
        else:
            os.mkdir(path)


def read_kitti_annotation(kitti_file):
    annotations = []
    for line in open(kitti_file):
        if len(line[:-1].split(' ')) == 8:
            [label, _, _, _, x1, y1, x2, y2] = line[:-1].split(' ')
        elif len(line[:-1].split(' ')) == 5:
            [label, x1, y1, x2, y2] = line[:-1].split(' ')
        else:
            raise Exception("Incompatible file !")
        annotations.append([label, float(x1), float(y1), float(x2), float(y2)])
    return annotations


def read_pascal_annotation(pascal_file):
    annotations = []
    tree = et.parse(pascal_file)
    label_main = tree.findall('object')
    coordinates = tree.findall('object/bndbox')
    for i in range(len(label_main)):
        text_label = label_main[i].find('name').text
        x1 = coordinates[i].find('xmin').text
        y1 = coordinates[i].find('ymin').text
        x2 = coordinates[i].find('xmax').text
        y2 = coordinates[i].find('ymax').text
        out_label = [text_label, float(x1), float(y1), float(x2), float(y2)]
        annotations.append(out_label)
    return annotations


def create_blank_image(width=320, height=240, rgb_color=(211, 211, 211)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)
    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


def calculate_area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)


def calculate_bbox_area(annotations):
    area_list = []
    for box in annotations:
        [x1, y1, x2, y2] = box[-4:]
        area_list.append(calculate_area(x1, y1, x2, y2))
    return area_list


class kitti_augmentation:
    def __init__(self, dataset_type, input_2class_path, augment_out_path, output_dimension, pixel_shift, visualize,
                 canvas_shift_percentage, brightness_percentage, contrast_percentage, canvas_shift,
                 brightness, contrast, apply_multiple_aug):
        start_time = time.time()
        self.input_data_root = input_2class_path
        self.Augment_out_root = augment_out_path
        self.out_dimension = [int(item) for item in output_dimension.split(',')]
        self.pixel_shift = int(pixel_shift)
        self.visualize_dir = "/tmp/visualize"
        self.visualize = visualize
        self.low_brightness = 0.7
        self.high_brightness = 1.3
        self.test_set_images_list = []
        self.is_test_set_available = False
        self.canvas_shift_percentage = canvas_shift_percentage
        self.brightness_percentage = brightness_percentage
        self.contrast_percentage = contrast_percentage
        self.dataset_type = dataset_type
        self.shift = canvas_shift
        self.brightness = brightness
        self.contrast = contrast
        self.apply_multiple_aug = apply_multiple_aug

        if dataset_type == 'kitti':
            # input dir structure
            self.input_image_set_path = os.path.join(self.input_data_root, "ImageSets")
            self.input_training_set_path = os.path.join(self.input_data_root, "training")
            self.input_images_path = os.path.join(self.input_training_set_path, "images")
            self.input_labels_path = os.path.join(self.input_training_set_path, "labels")
            self.input_image_set_file = os.path.join(self.input_image_set_path, 'train.txt')
            self.input_test_set_file = os.path.join(self.input_image_set_path, 'test.txt')

            # output dir structure
            self.image_set_path = os.path.join(self.Augment_out_root, "ImageSets")
            self.training_set_path = os.path.join(self.Augment_out_root, "training")
            self.images_path = os.path.join(self.training_set_path, "images")
            self.labels_path = os.path.join(self.training_set_path, "labels")
            self.testing_set_path = os.path.join(self.Augment_out_root, "testing")
            self.test_images_path = os.path.join(self.testing_set_path, "images")
            self.test_labels_path = os.path.join(self.testing_set_path, "labels")

            self.path_list = [self.Augment_out_root, self.image_set_path, self.training_set_path,
                              self.images_path, self.labels_path, self.testing_set_path,
                              self.test_images_path, self.test_labels_path]

        elif dataset_type == 'pascal':
            # pascal input dir structure
            self.input_images_path = os.path.join(self.input_data_root, "JPEGImages")
            self.input_labels_path = os.path.join(self.input_data_root, "Annotations")
            self.input_image_set = os.path.join(self.input_data_root, "ImageSets")
            self.input_image_set_path = os.path.join(self.input_image_set, "Main")
            self.input_image_set_file = os.path.join(self.input_image_set_path, 'train.txt')
            self.input_test_set_file = os.path.join(self.input_image_set_path, 'test.txt')

            # pascal output dir structure
            self.input_image_set_path = os.path.join(self.Augment_out_root, "Main")
            self.image_set_path = os.path.join(self.Augment_out_root, "ImageSets")
            self.images_path = os.path.join(self.Augment_out_root, "JPEGImages")
            self.labels_path = os.path.join(self.Augment_out_root, "Annotations")
            self.testing_set_path = os.path.join(self.Augment_out_root, "testing")
            self.test_images_path = os.path.join(self.testing_set_path, "images")
            self.test_labels_path = os.path.join(self.testing_set_path, "labels")

            self.path_list = [self.Augment_out_root, self.image_set_path, self.input_image_set_path,
                              self.images_path, self.labels_path, self.testing_set_path,
                              self.test_images_path, self.test_labels_path]
        else:
            raise Exception("Invalid Dataset type !")

        if os.path.exists(self.input_test_set_file):
            self.is_test_set_available = True
            for test_image in open(self.input_test_set_file):
                self.test_set_images_list.append(test_image[:-1])

        self.image_set_file = os.path.join(self.image_set_path, 'train.txt')

        # validate_annotations
        self.validate_annotations()

        if self.visualize:
            self.path_list.append(self.visualize_dir)
        create_dir(self.path_list)

        self.class_colors = [(0, 255, 255)]

        self.invalid_box_count = 0

        self.cpu_count = multiprocessing.cpu_count()
        self.process_pool = multiprocessing.Pool(int(self.cpu_count * 3 / 4))
        self.contrast_choices = [1.5, 2, 2.5]
        self.augment()

        self.process_pool.close()
        self.process_pool.join()

        with open(self.image_set_file, 'a+') as imageSet:
            for name in os.listdir(self.labels_path):
                imageSet.write(name[:-4] + "\n")

        print("Time Taken : {}".format(time.time() - start_time))
        print("==" * 50, "\nAugmentation operation finished\n")
        print("Total Discarded invalid boxes :{}".format(self.invalid_box_count))
        print("==" * 50, "\n\n")

    def validate_annotations(self):

        if not os.path.isfile(self.input_image_set_file) or not os.path.isdir(self.input_labels_path) \
                or not os.path.isdir(self.input_images_path):
            error(-3)
        labels = os.listdir(self.input_labels_path)
        images = os.listdir(self.input_images_path)
        if not (len(labels) == len(images)):
            error(-3)

    @staticmethod
    def validate_box(image, annotations):
        image_height, image_width, c = image.shape
        for bbox in annotations:
            box = bbox[-4:]
            if (box[2] - box[0] <= 10) or (box[3] - box[1] <= 10):
                return False
            if box[0] < 0 or box[1] < 0 or box[2] > image_width or box[3] > image_height or box[2] <= 0 or bbox[3] <= 0:
                return False
        return True

    def brightness_aug(self, img, name, annotations, from_canvas_aug=False):
        if from_canvas_aug:
            canvas, annotations_canvas, base_name = img.copy(), annotations.copy(), name
        else:
            canvas, annotations_canvas, base_name = self.add_canvas_set(img, annotations, name)
        base_name = name
        for brightness_value in np.linspace(self.low_brightness, self.high_brightness, 3):
            # brightness_value = 0.8
            hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 1] = hsv[:, :, 1] * brightness_value
            hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
            hsv[:, :, 2] = hsv[:, :, 2] * brightness_value
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
            hsv = np.array(hsv, dtype=np.uint8)
            canvas = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            aug_name = base_name + "br-{}".format(brightness_value)
            self.write_output(canvas, aug_name, annotations_canvas.copy())

    def contrast_aug(self, img, name, annotations, from_canvas_aug=False):
        if from_canvas_aug:
            canvas, annotations_canvas, base_name = img.copy(), annotations.copy(), name
        else:
            canvas, annotations, base_name = self.add_canvas_set(img, annotations, name)
        # for con in self.contrast_choices:
        con = random.choice(self.contrast_choices)
        adjusted = cv2.convertScaleAbs(canvas, alpha=con)
        self.write_output(adjusted, "{}_con_{}".format(name, con), annotations.copy())

    def write_output(self, canvas, name, annotations):
        out_annotations = annotations.copy()
        if self.visualize:
            self.visualize_bbox(canvas, out_annotations, name)
        if not self.validate_box(canvas, out_annotations):
            self.invalid_box_count += 1
            return

        output_image_path = os.path.join(self.images_path, name + '.jpg')
        output_label_path = os.path.join(self.labels_path, name + '.txt')
        cv2.imwrite(output_image_path, canvas)

        for label in out_annotations:
            label_copy = label.copy()
            label_copy.insert(1, "0.0 0.0 0.0")
            with open(output_label_path, "a+") as file:
                kitti_format = ""
                for element in label_copy:
                    kitti_format += str(element) + " "
                file.write(kitti_format.strip() + "\n")

    def canvas_shift_aug(self, image, annotations, base_name):
        height, width, channels = image.shape
        width_to_height = width / height
        if height > width * 3:
            return
        if width > height:
            width_resize, height_resize = self.out_dimension[0], int(self.out_dimension[1] / width_to_height)
            width_scale, height_scale = width_resize / width, height_resize / height
            image = cv2.resize(image, (width_resize, height_resize))
            for shift in range(0, self.out_dimension[1] - height_resize, self.pixel_shift):
                out_annotations = []
                canvas = create_blank_image(width=self.out_dimension[0], height=self.out_dimension[1])
                canvas[shift:height_resize + shift, 0:width_resize] = image[0:height_resize, 0:width_resize]
                name = '{}{}'.format(base_name, str(shift))
                for box in annotations:
                    out_annotations.append([box[0], float(box[1]) * width_scale,
                                            float(box[2]) * height_scale + shift,
                                            float(box[3]) * width_scale,
                                            float(box[4]) * height_scale + shift
                                            ])
                self.write_output(canvas, name, out_annotations)
                if self.apply_multiple_aug:
                    if self.brightness:
                        self.brightness_aug(canvas.copy(), name, out_annotations.copy(), from_canvas_aug=True)
                    if self.contrast:
                        self.contrast_aug(canvas.copy(), name, out_annotations.copy(), from_canvas_aug=True)

        else:
            width_resize, height_resize = int(self.out_dimension[0] * width_to_height), self.out_dimension[1]
            width_scale, height_scale = width_resize / width, height_resize / height
            image = cv2.resize(image, (width_resize, height_resize))
            for shift in range(0, self.out_dimension[0] - width_resize, self.pixel_shift):
                out_annotations = []
                canvas = create_blank_image(width=self.out_dimension[0], height=self.out_dimension[1])
                canvas[0:height_resize, shift:width_resize + shift] = image[0:height_resize, 0:width_resize]
                name = '{}{}'.format(base_name, str(shift))
                for box in annotations:
                    out_annotations.append([box[0], float(box[1]) * width_scale + shift,
                                            float(box[2]) * height_scale,
                                            float(box[3]) * width_scale + shift,
                                            float(box[4]) * height_scale
                                            ])
                self.write_output(canvas, name, out_annotations)
                if self.apply_multiple_aug:
                    if self.brightness:
                        self.brightness_aug(canvas.copy(), name, out_annotations.copy(), from_canvas_aug=True)
                    if self.contrast:
                        self.contrast_aug(canvas.copy(), name, out_annotations.copy(), from_canvas_aug=True)

    def add_canvas_set(self, image, annotations, base_name):
        height, width, channels = image.shape
        width_to_height = width / height
        if width > height:
            width_resize, height_resize = self.out_dimension[0], int(self.out_dimension[1] / width_to_height)
            width_scale, height_scale = width_resize / width, height_resize / height
            image = cv2.resize(image, (width_resize, height_resize))
            shift = int((self.out_dimension[1] - height_resize) / 2)
            out_annotations = []
            canvas = create_blank_image(width=self.out_dimension[0], height=self.out_dimension[1])

            canvas[shift:height_resize + shift, 0:width_resize] = image[0:height_resize, 0:width_resize]
            name = '{}'.format(base_name)
            for box in annotations:
                out_annotations.append([box[0], float(box[1]) * width_scale,
                                        float(box[2]) * height_scale + shift,
                                        float(box[3]) * width_scale,
                                        float(box[4]) * height_scale + shift
                                        ])
        else:
            width_resize, height_resize = int(self.out_dimension[0] * width_to_height), self.out_dimension[1]
            width_scale, height_scale = width_resize / width, height_resize / height
            image = cv2.resize(image, (width_resize, height_resize))
            shift = int((self.out_dimension[0] - width_resize) / 2)
            out_annotations = []
            canvas = create_blank_image(width=self.out_dimension[0], height=self.out_dimension[1])
            canvas[0:height_resize, shift:width_resize + shift] = image[0:height_resize, 0:width_resize]
            name = '{}{}'.format(base_name, str(shift))
            for box in annotations:
                out_annotations.append([box[0], float(box[1]) * width_scale + shift,
                                        float(box[2]) * height_scale,
                                        float(box[3]) * width_scale + shift,
                                        float(box[4]) * height_scale
                                        ])
        return canvas, out_annotations, name

    def add_canvas_test_set(self, image, annotations, base_name):
        if self.visualize:
            self.visualize_bbox(image, annotations, base_name)
        canvas, annotations, name = self.add_canvas_set(image, annotations, base_name)
        # self.write_output(canvas, name, annotations)
        # print(name)
        output_image_path = os.path.join(self.test_images_path, name + '.jpg')
        output_label_path = os.path.join(self.test_labels_path, name + '.txt')
        cv2.imwrite(output_image_path, canvas)
        for label in annotations:
            label_copy = label.copy()
            label_copy.insert(1, "0.0 0.0 0.0")
            with open(output_label_path, "a+") as file:
                kitti_format = ""
                for element in label_copy:
                    kitti_format += str(element) + " "
                file.write(kitti_format.strip() + "\n")

    def visualize_bbox(self, image, annotations, base_name):
        image = image.copy()
        output_image_path = os.path.join(self.visualize_dir, base_name + '.jpg')
        for label in annotations:
            [x1, y1, x2, y2] = label[-4:]
            class_index = 0
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), self.class_colors[class_index], thickness=2)
        cv2.imwrite(output_image_path, image)

    def read_data(self, base_name):
        if self.dataset_type == "kitti":
            label_path = os.path.join(self.input_labels_path, base_name + '.txt')
            img_path = os.path.join(self.input_images_path, base_name + '.jpg')
            if os.path.exists(label_path) and os.path.exists(img_path):
                annotations = read_kitti_annotation(label_path)
                image = cv2.imread(img_path)
                return annotations, image
            else:
                print("Label or image is not found for {} {}".format(label_path, img_path))
                sys.exit()
        elif self.dataset_type == "pascal":
            label_path = os.path.join(self.input_labels_path, base_name + '.xml')
            img_path = os.path.join(self.input_images_path, base_name + '.jpg')
            if os.path.exists(label_path) and os.path.exists(img_path):
                annotations = read_pascal_annotation(label_path)
                image = cv2.imread(img_path)
                return annotations, image
            else:
                print("Label or image is not found for {} {}".format(label_path, img_path))
                sys.exit()

    def augment(self):
        images_name_list = os.listdir(self.input_images_path)
        random.shuffle(images_name_list)
        len_total_images = len(images_name_list)
        len_images = len_total_images - len(self.test_set_images_list)
        if self.canvas_shift_percentage > 100 or self.brightness_percentage > 100 or self.contrast_percentage > 100:
            error(-7)
        if self.shift:
            shift_aug_image_size = int((len_images * self.canvas_shift_percentage) / 100)
        else:
            shift_aug_image_size = 0
            self.canvas_shift_percentage = 0
        if self.brightness:
            brightness_aug_image_size = int((len_images * self.brightness_percentage) / 100)
        else:
            brightness_aug_image_size = 0
            self.brightness_percentage = 0
        if self.contrast:
            contrast_aug_image_size = int((len_images * self.contrast_percentage) / 100)
        else:
            contrast_aug_image_size = 0
            self.contrast_percentage = 0
        total_percentage = self.canvas_shift_percentage + self.brightness_percentage + self.contrast_percentage

        self.shift_aug_targets = []
        self.brightness_aug_targets = []
        self.contrast_aug_targets = []

        total_images_for_split = images_name_list.copy() + images_name_list.copy() + images_name_list.copy()

        self.shift_aug_targets = total_images_for_split[:shift_aug_image_size]
        brightness_end_index = shift_aug_image_size + brightness_aug_image_size
        self.brightness_aug_targets = total_images_for_split[shift_aug_image_size:brightness_end_index]
        contrast_end_index = brightness_end_index + contrast_aug_image_size
        self.contrast_aug_targets = total_images_for_split[brightness_end_index:contrast_end_index]

        total_aug_images = len(self.test_set_images_list) + len(self.shift_aug_targets) + len(
            self.brightness_aug_targets) + len(self.contrast_aug_targets)
        input_images_count = 0
        with tqdm(total=total_aug_images, file=sys.stdout) as pbar:
            for line in self.test_set_images_list:
                # base_name = line[:-4]
                annotations, image = self.read_data(line)
                self.add_canvas_test_set(image, annotations, line)
                pbar.update(1)

            for line in self.shift_aug_targets:
                base_name = line[:-4]
                annotations, image = self.read_data(base_name)
                self.canvas_shift_aug(image, annotations, base_name)
                # self.add_canvas_test_set(image, annotations, line)
                input_images_count += 1
                pbar.update(1)

            for line in self.brightness_aug_targets:
                base_name = line[:-4]
                annotations, image = self.read_data(base_name)
                self.brightness_aug(image, base_name, annotations)
                input_images_count += 1
                pbar.update(1)

            for line in self.contrast_aug_targets:
                base_name = line[:-4]
                annotations, image = self.read_data(base_name)
                self.contrast_aug(image, base_name, annotations)
                pbar.update(1)
                input_images_count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input Dataset Path")
    parser.add_argument("-o", "--output", required=True, help="Output Dataset Path")
    parser.add_argument("-cs", "--canvas_shift", action="store_true",
                        help="whether we want to do shift augmentation ")
    parser.add_argument("-b", "--brightness", action="store_true",
                        help="whether we want to do brightness augmentation")
    parser.add_argument("-c", "--contrast", action="store_true",
                        help="whether we want to do contrast augmentation")
    parser.add_argument("-s", "--pixel_shift", required=False, default=5, help="Canvas augmentation pixel shift. "
                                                                                "(only for augmentation)")
    parser.add_argument("-t", "--type", default='kitti', help="Type of Dataset kitti/pascal")
    parser.add_argument("-d", "--output_dimension", required=False, default='64,64',
                        help="Output image dimension in terms of x,y")
    parser.add_argument("-m", "--apply_multiple_aug", action="store_true", help="If user wants to apply another"
                                                                                " augmentation in already augmented image")
    parser.add_argument("-v", "--visualize", action="store_true", help="If user wants to visualize"
                                                                       "augmented annotations")
    parser.add_argument("-csp", "--canvas_shift_percentage", required=False, default=100,
                        help="Percentage required for canvas_augmentation")
    parser.add_argument("-bp", "--brightness_percentage", required=False, default=100,
                        help="Percentage required for brightness_augmentation")
    parser.add_argument("-cp", "--contrast_percentage", required=False, default=100,
                        help="Percentage required for contrast_augmentation")
    args = parser.parse_args()

    kitti_augmentation(args.type, args.input, args.output, args.output_dimension, args.pixel_shift, args.visualize,
                       int(args.canvas_shift_percentage), int(args.brightness_percentage),
                       int(args.contrast_percentage),
                       args.canvas_shift, args.brightness, args.contrast, args.apply_multiple_aug)
