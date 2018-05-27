## serving tensorflow .pb
https://github.com/matterport/Mask_RCNN/issues/218

To produce h5:
python3 coco.py evaluate --dataset=$COCO_PATH --model=coco

To save model in coco.py:
evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit)) model.keras_model.save("mrcnn_eval.h5")

Extracting pb from h5:
python3 keras_to_tensorflow.py -input_model_file mask_rcnn_numberplate.h5 -output_model_file mask_rcnn_numberplate.pb -num_outputs=7

python keras_to_tensorflow.py -input_model_file mask_rcnn_numberplate.h5 -output_model_file mask_rcnn_numberplate.pb

Could you paste the full stacktrace?

权重文件不行的，必须要带有网络结构。就是说，被转换的h5文件应该是 keras中的 model.save()，而不是model.save_weights()



server.py
manager.py
api_call.py
想走多线程路线，结果不行，tensorflow不支持线程之间共享model
