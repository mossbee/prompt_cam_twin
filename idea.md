This is idea.md, a markdown file inside the official implementation of the research PROMPT-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis. The whole research is is research.md.

Purpose:
- Adapt the idea of Prompt-Cam to a similar task: identical twin faces verification (given two highly similar face image, tell whether they are same person or not). Twin faces look highy similar, except for small details - which can be consider as an fine-grained image analysis task.

Dataset for training:
- Dataset for training is the ND TWIN 2009 - 2010 dataset (captured at the Twins Days Festivals in Twinsburg, Ohio in 2009 and 2010). Faces images were pre-processed through: face landmarks detection, face alignment, face detection, cropped out face-only images. Just tell me the size to resize to, no matter square ratio or rectangle, a dataset is ready.
- The dataset are already splitted into train and test sets. A file called train_dataset_infor.json in data folder captures the structure of train dataset under a dictionary, with keys are person ids and corresponding value to each key is a list of image paths. A train_twin_pairs.json in the same folder shows the information on twin pairs under a list of twin person id pairs. The same with test set: test_dataset_infor.json and test_twin_pairs.json.
- The smallest number of images per person is 4, biggest is 68, total image are 6182, total number of person is 356 (178 twin pairs). Test set has 62 ids - 31 twin pairs, 907 images. Take a look at some of their first line for more information.

Training resources:
- Local Ubuntu server with no data uploading ability (security) with two 2080Ti GPU(cuda:0, cuda:1)
- Kaggle (2 Nvidia Tesla T4 or Nvidia Tesla P100).

Tracking method:
- The Ubuntu server has a locally hosted MLFlow service for training tracking. Just plug and play.
- For Kaggle, API key is in Kaggle kaggle secrets, in WANDB_API_KEY variable. The entity name is hunchoquavodb-hanoi-university-of-science-and-technology
- I want three training tracking method: MLFlow, WanDB and no tracking.

More information:
- Kaggle training will time out every 12 hours, so model checkpoint should be performed every epoch, and the training should be able to resume from a checkpoint.

Question:
- Is the idea of Prompt-CAM suitble for my task? (Consider the downstream task characteristic, training data size, etc)
- In the original paper, author can show the traits that differ twin people through attention map (both image are applied an attention map onto, which part discriminative among two images will brighter). Can we have that ability on identical twin verification task? (Show which details in two faces are differ - maybe the eye, nose etc)
- If you have any question, ask me.

Things to do:
- Write an implementation plan markdown file, for YOU to revisit while implement the idea. This file's purpose is to prevent you from forget what you are doing. It should be concise, clear but abstract about project structure, what you are writing in this files, etc...
- And also leverage the exist codebase, just implement what needed.
- I want a code base that efficient, clear, just what needed. The code base should fully utilized training resources.