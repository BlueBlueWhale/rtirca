import argparse

from ultralytics.models.yolo.model import YOLO
from ultralytics.models.rtirca.train import RTIRCATrainer
from exp.config.config import MODEL_STRUCTURES, MODEL_FILES


# Parse command line arguments
parser = argparse.ArgumentParser(description="Run MLKD training")
parser.add_argument("--data", required=True, help="Dataset configuration file")
parser.add_argument("-s", "--student", required=True, help="Student model name")
parser.add_argument("-t", "--teacher", required=True, help="Teacher model name")
parser.add_argument("-tc", "--teacher-ckpt", required=True, help="Path to the trained teacher checkpoint")
args = parser.parse_args()

# Load the student model
student = YOLO(MODEL_FILES[args.student]["ckpt"])

# Perform MLKD training
results = student.train(
    trainer=RTIRCATrainer,
    cfg="config/cfg.yaml",
    data=args.data,
    activation_layers=MODEL_STRUCTURES[args.student]["activation_layers"],
    student_channels=MODEL_STRUCTURES[args.student]["output_channels"],
    teacher_channels=MODEL_STRUCTURES[args.teacher]["output_channels"],
    teacher_ckpt=args.teacher_ckpt,
)
