# Conversational AI Training and Chat

This project, developed by Bhuvan Indra G, trains a conversational AI based on DistilGPT2, fine-tuned on the `pippa.jsonl` dataset from Hugging Face, and includes a script to chat with the model. Created for a college portfolio to demonstrate machine learning skills, it was tested on Kaggle Notebooks with GPU P100. The project is licensed under the MIT License, allowing free use while disclaiming liability (see [LICENSE](LICENSE.txt) and [DISCLAIMER](DISCLAIMER.md)).

## Data Attribution

The `pippa.jsonl` dataset is sourced from [PygmalionAI/PIPPA](https://huggingface.co/datasets/PygmalionAI/PIPPA) on Hugging Face. All usage complies with their terms, and credit is given to PygmalionAI for providing the dataset. Note that the dataset contains content that may not be suitable for all audiences; refer to the dataset card for details.

## Project Status

The chatbot’s responses may not always be coherent due to limitations in the dataset and model size. Contributions to improve coherence or functionality are welcome! Please reach out via [Contact](#contact) to collaborate.

## Setup

1. **Prerequisites**:
   - Kaggle account with phone verification.
   - Internet enabled.

2. **Enable GPU**:
   - Create a Python notebook on Kaggle.
   - Set Accelerator to “GPU P100.”

3. **Download Dataset**:
   ```bash
   wget -O /kaggle/working/pippa.jsonl "https://huggingface.co/datasets/PygmalionAI/PIPPA/resolve/main/pippa.jsonl"
   ```

4. **Train Model**:
   ```bash
   python train_chatbot_kaggle.py
   ```

5. **Chat with Model**:
   ```bash
   python chat_with_model.py
   ```

## Training Parameters

- Dataset: Up to 10,000 conversations.
- Epochs: 3.
- Batch Size: 4 (effective 16 with accumulation).
- Max Length: 512.
- Learning Rate: 5e-5 with cosine scheduler and warmup.
- Early Stopping: Patience 1 epoch.

## Kaggle Notes

- **Environment**: Code was developed and tested on Kaggle Notebooks with GPU P100 (16GB VRAM, 30 hours/week quota). No additional dependencies were installed, as Kaggle’s environment includes required libraries.
- **Runtime**: ~1–2 hours for training.
- **Troubleshooting**:
  - Memory: Reduce `BATCH_SIZE=2` or `MAX_LENGTH=256` in `train_chatbot_kaggle.py` if CUDA errors occur.
  - GPU: Verify `torch.cuda.is_available()` in a code cell.
  - Dataset: Ensure `pippa.jsonl` is downloaded correctly.

## Disclaimer

This project is for educational purposes only. Please read [DISCLAIMER](DISCLAIMER.md) for important information on liability, dataset content, and precautions before running the code.

## License

This project is licensed under the MIT License (see [LICENSE](LICENSE.txt)), allowing anyone to use, modify, or distribute the code, subject to the terms therein. DistilGPT2 is licensed under Apache 2.0 by Hugging Face.

## Attribution

- DistilGPT2: [Hugging Face](https://huggingface.co/distilgpt2)
- Dataset: [PygmalionAI/PIPPA](https://huggingface.co/datasets/PygmalionAI/PIPPA)

## Contact

To contribute to the project or discuss improvements, contact Bhuvan Indra G via:
- [Email](mailto:gbindra21@gmail.com)
- [Instagram](https://www.instagram.com/bhuvan_indra_0520/)
- [LinkedIn](https://www.linkedin.com/in/bhuvan-indra-995828274?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
- Phone no: +91 9491149955
