# Disclaimer

This project, including all code, models, and documentation, is provided for educational purposes as part of a college portfolio to demonstrate machine learning skills. Developed by Bhuvan Indra G, it is intended to showcase technical abilities and is not for commercial use. By using this project, you agree to the following terms and acknowledge the associated risks.

## No Liability for Damages

The author (Bhuvan Indra G) is not responsible for any damages, including but not limited to hardware damage, software issues, data loss, or other adverse effects, that may result from running the code or using the trained model. Users run the code at their own risk and should take appropriate precautions, such as:

- **Using a suitable environment**: The code was developed and tested on Kaggle Notebooks with GPU P100 (16GB VRAM). Running on personal laptops or other hardware may lead to overheating, memory issues, or other problems.
- **Monitoring resources**: Check GPU/CPU memory usage (`nvidia-smi`) and ensure adequate cooling and power supply.
- **Backing up data**: Avoid running the code on systems with critical data without backups.

## Dataset Content Warning

The `pippa.jsonl` dataset is sourced from [PygmalionAI/PIPPA](https://huggingface.co/datasets/PygmalionAI/PIPPA) on Hugging Face and is not owned or created by the author. Per the dataset card, it contains content that may not be suitable for all audiences, including potentially offensive or sensitive material. The author has implemented content filtering to skip explicit content, but no guarantees are made about the appropriateness of the dataset or model outputs. Users should:

- Review the dataset card before use.
- Avoid using the model in production or sensitive applications without thorough testing.
- Be aware that model responses may reflect biases or inappropriate content from the dataset.

## No Warranty

The code and model are provided “as is” without any warranty, express or implied, including but not limited to fitness for a particular purpose or non-infringement. The author does not guarantee the accuracy, safety, or reliability of the model’s outputs.

## Precautions for Contact

To minimize risks when running this project:

- **Use cloud platforms**: Prefer Kaggle Notebooks or Google Colab with GPU support over personal hardware.
- **Verify compatibility**: Ensure your system meets the requirements (Python 3.8+, dependencies in `requirements.txt`).
- **Test in a sandbox**: Run the code in a virtual environment or isolated system to avoid conflicts.
- **Monitor execution**: Stop training if memory usage exceeds safe limits or if hardware shows signs of strain.
- **Avoid sensitive inputs**: Do not input personal or sensitive data into the chat script, as the model may generate unpredictable responses.
- **Check dataset integrity**: Verify `pippa.jsonl` is downloaded from the official Hugging Face link to avoid corrupted files.

## Contact

For questions, issues, or contributions, refer to the [Contact](#contact) section in [README.md](README.md). The author is not obligated to provide support or updates.

By using this project, you acknowledge that you have read and understood this disclaimer and agree to use the code responsibly.