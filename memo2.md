## added into calpis water detection mask r-cnn.ipynb

file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())


{
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]

# filename = os.path.join(IMAGE_DIR, '04H0257.jpg')
filename = os.path.join(IMAGE_DIR, random.choice(file_names))
image = skimage.io.imread(filename)

base_file_name = os.path.basename(filename)
base_dir_name = os.path.dirname(filename)
print('dir name:',base_dir_name)
print('file name:',base_file_name)
split_file_name, split_file_ext = os.path.splitext(base_file_name)
print(split_file_name)
print(split_file_ext)

img_timestamp = time.localtime()
time_string = str.join('',(str(img_timestamp.tm_year), str(img_timestamp.tm_mon).zfill(2), str(img_timestamp.tm_mday).zfill(2)))
print(time_string)

saved_dir_name = 'masked'
saved_file_name = str.join('\\', (base_dir_name, saved_dir_name, base_file_name))
print(saved_file_name)

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                            class_names, r['scores'])
print_img= visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])

skimage.io.imsave(saved_file_name,print_img)
}



###################################33
load every file in dest folder then run detection
{
  # Load every image from the images folder

  image_type_ok_list = ['jpeg','png','gif','bmp']

  for file_names in os.scandir(IMAGE_DIR):  
      if file_names.is_file():
          image_type = imghdr.what(file_names)
          if image_type in image_type_ok_list:            
              filename = os.path.join(IMAGE_DIR, file_names)
              print('Loading image:',file_names)

              image = skimage.io.imread(filename)
              if len(image.shape) > 2 and image.shape[2] == 4:
                  image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

              base_file_name = os.path.basename(filename)
              base_dir_name = os.path.dirname(filename)
              print('dir name:',base_dir_name)
              print('file name:',base_file_name)
              split_file_name, split_file_ext = os.path.splitext(base_file_name)
              print(split_file_name)
              print(split_file_ext)

              img_timestamp = time.localtime()
              time_string = str.join('',(str(img_timestamp.tm_year), str(img_timestamp.tm_mon).zfill(2), str(img_timestamp.tm_mday).zfill(2)))
              print(time_string)

              saved_dir_name = 'masked'
              saved_file_name = str.join('\\', (base_dir_name, saved_dir_name, base_file_name))
              print(saved_file_name)

              # Run detection
              results = model.detect([image], verbose=1)

              # Visualize results
              r = results[0]
              # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
              #                            class_names, r['scores'])
              print_img= visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                          class_names, r['scores'])

              skimage.io.imsave(saved_file_name,print_img)
}
