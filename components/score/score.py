import argparse
from pathlib import Path
from sklearn.metrics import classification_report,mean_squared_error, r2_score
import os
import mlflow
import pandas as pd

mlflow.sklearn.autolog()

parser = argparse.ArgumentParser("score")
parser.add_argument("--model_input", type=str, help="Path of input model")
parser.add_argument("--test_data", type=str, help="Path to test data")
parser.add_argument("--score_output", type=str, help="Path of scoring output")

args = parser.parse_args()

print("hello scoring world...")

lines = [
    f"Model path: {args.model_input}",
    f"Test data path: {args.test_data}",
    f"Scoring output path: {args.score_output}",
]

for line in lines:
    print(line)

# Load the model from input port 4
pathmodel = os.path.join(args.model_input, "trained_model")
model = mlflow.sklearn.load_model(pathmodel)

# testfiles = os.listdir(args.test_data)
# os.path.join(args.test_data, testfiles[0])
# paths are mounted as folder, therefore, we are selecting the file from folder
test_df = pd.read_excel(args.test_data, header=1, index_col=0)
# test_df = pd.read_csv(os.path.join(args.test_data, testfiles[0]))

# Extracting the label column
y_test = test_df.pop("default payment next month")

# convert the dataframe values to array
X_test = test_df.values

y_pred = model.predict(X_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
print("Model: ", model)

# Print the results of scoring the predictions against actual values in the test data
# The coefficients
# print("Coefficients: \n", model)

# # Load the model from input port
# # Here only print the model as text since it is a dummy one
# model = (Path(args.model_input) / "model.txt").read_text()
# print("Model: ", model)

# # Do scoring with the input model
# # Here only print text to output file as demo
(Path(args.score_output) / "score.txt").write_text(
    "Scored with the following mode:\n{}".format(model)
)
