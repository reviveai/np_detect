## serving tensorflow .pb
https://github.com/matterport/Mask_RCNN/issues/218

To produce h5:
python3 coco.py evaluate --dataset=$COCO_PATH --model=coco

To save model in coco.py:
evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit)) model.keras_model.save("mrcnn_eval.h5")

Extracting pb from h5:
python3 keras_to_tensorflow.py -input_model_file saved_model_mrcnn_eval.h5 -output_model_file model.pb -num_outputs=7

Could you paste the full stacktrace?
