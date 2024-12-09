import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import zipfile
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from skimage.feature import local_binary_pattern

# Define the paths for the uploaded files
zip_file_path = '/content/Images.zip'

# Create a directory to extract the contents
extract_dir = 'Images'
os.makedirs(extract_dir, exist_ok=True)

# Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Update the path to the images directory
images_path = os.path.join(extract_dir, 'Images')

# Define the path to the folder containing the images
folder_dir = 'Images/Images'
image_files = []

# Traverse the directory tree recursively
for root, dirs, files in os.walk(folder_dir):
    for file in files:
        if file.lower().endswith(".jpg"):
            image_path = os.path.join(root, file)
            image_files.append((file, image_path))

# Create a DataFrame from the extracted image files
image_df = pd.DataFrame(image_files, columns=['Image', 'image_path'])

# Display the first few rows of the image DataFrame
print("Image DataFrame:")
print(image_df.head())

# Load the train and test DataFrames
train = pd.read_csv('/content/train2.csv')
test = pd.read_csv('test.csv')

# Adjust the 'Image' column in train and test DataFrames
train['Image'] = train['Image'].apply(lambda x: f"BloodImage_{int(x):05d}.jpg")
test['Image'] = test['Image'].apply(lambda x: f"BloodImage_{int(x):05d}.jpg")

# Display the first few rows of the train DataFrame
print("Train DataFrame:")
print(train.head())

# Display the first few rows of the test DataFrame
print("Test DataFrame:")
print(test.head())

# Merge the train DataFrame with image_df on the "Image" column
train = pd.merge(train, image_df, on='Image', how='left')

# Filter out rows with NaN image paths
train = train.dropna(subset=['image_path'])

# Display the first few rows of the updated train DataFrame
print("Train Dataset with Image Paths:")
print(train.head())

# Merge the test DataFrame with image_df on the "Image" column
test = pd.merge(test, image_df, on='Image', how='left')

# Filter out rows with NaN image paths
test = test.dropna(subset=['image_path'])

# Display the first few rows of the updated test DataFrame
print("Test Dataset with Image Paths:")
print(test.head())

# Define the segmentation function
def segment_image(image_path, output_dir):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for the color of the contour (adjust these values as needed)
    lower_color = np.array([100, 50, 50])  # Example lower bound for purple
    upper_color = np.array([140, 255, 255])  # Example upper bound for purple

    # Create a mask based on the color range
    mask = cv2.inRange(hsv_image, lower_color, upper_color)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Select the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a mask for the largest contour
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Extract the region of interest (ROI) using the mask
        roi = cv2.bitwise_and(image_rgb, image_rgb, mask=contour_mask)

        # Create a new blank image with the same dimensions as the original
        blank_image = np.zeros_like(image_rgb)

        # Find the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Paste the ROI onto the new blank image
        blank_image[y:y+h, x:x+w] = roi[y:y+h, x:x+w]

        # Save the segmented image
        segmented_image_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(segmented_image_path, cv2.cvtColor(blank_image, cv2.COLOR_RGB2BGR))

        return segmented_image_path
    return None

# Create a directory for the segmented images
segmented_dir = 'Segmented_Images'
os.makedirs(segmented_dir, exist_ok=True)

# Apply the segmentation function to each image in the dataset
train['segmented_image_path'] = train['image_path'].apply(lambda x: segment_image(x, segmented_dir))
test['segmented_image_path'] = test['image_path'].apply(lambda x: segment_image(x, segmented_dir))

# Display the first few rows of the updated train DataFrame
print("Train Dataset with Segmented Image Paths:")
print(train.head())

# Display the first few rows of the updated test DataFrame
print("Test Dataset with Segmented Image Paths:")
print(test.head())

# Define the function to display segmented images
def display_segmented_images(data, num_images=9):
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    for i, (idx, row) in enumerate(data.tail(num_images).iterrows()):
        image = cv2.imread(row['segmented_image_path'])
        ax[i // 3, i % 3].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[i // 3, i % 3].axis('off')
        ax[i // 3, i % 3].set_title(f"Label: {row['Category']}")
    plt.tight_layout()
    plt.show()

train.head()

test.head()

train.info()

!pip install sweetviz

# prompt: creat a copy of train

train_copy = train.copy()


import sweetviz as sv
import pandas as pd
import numpy as np

# Convert NumPy arrays in relevant columns to hashable types
for col in train.columns:
    if train[col].dtype == 'object':  # Check if the column contains objects
        try:
            train[col] = train_copy[col].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) else x)
        except:
            pass  # Handle cases where conversion might not be applicable

# Convert 'lbp_features' column to string type
train['lbp_features'] = train['lbp_features'].astype(str)  # Convert to string

import sweetviz as sv

my_report = sv.analyze(train)
my_report.show_html() # Default arguments will generate to "SWEETVIZ_REPORT.html"



# Call the function with the train data
display_segmented_images(train)

# Display segmented images of the test dataset
fig, ax = plt.subplots(3, 3, figsize=(15, 15))
for i in range(9):
    image_path = test['segmented_image_path'].iloc[i]
    image = cv2.imread(image_path)
    ax[i // 3, i % 3].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[i // 3, i % 3].axis('off')
plt.tight_layout()
plt.show()

# Define the function to extract LBP features
def extract_lbp_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize the histogram
    return hist

train['lbp_features'] = train['segmented_image_path'].apply(lambda x: extract_lbp_features(x))
test['lbp_features'] = test['segmented_image_path'].apply(lambda x: extract_lbp_features(x))

# Convert the lists of features into a DataFrame
X_train = pd.DataFrame(train['lbp_features'].tolist())
X_test = pd.DataFrame(test['lbp_features'].tolist())
y_train = train['Category']

# Encode the labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# Display the first few rows of the feature DataFrame
print("LBP Features DataFrame (Train):")
print(X_train.head())

# Apply PCA
n_components = min(len(X_train), X_train.shape[1])
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("PCA-transformed features (Train):")
print(X_train_pca[:5])

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_pca)
X_test_scaled = scaler.transform(X_test_pca)

# Define the parameter grid
param_grid = {
    'knn__n_neighbors': [3, 5],
    'knn__weights': ['uniform'],
    'knn__metric': ['euclidean'],
    'xgb__n_estimators': [50, 100],
    'xgb__learning_rate': [0.1],
    'xgb__max_depth': [3, 5],
    'svm__C': [1, 10],
    'svm__kernel': ['linear', 'rbf']
}

# Define the stacking classifier
stacking_clf = StackingClassifier(
    estimators=[
        ('knn', KNeighborsClassifier()),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')),
        ('svm', SVC(probability=True))
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

# Initialize Grid Search
grid_search = GridSearchCV(estimator=stacking_clf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)

# Train the model on the whole training data
grid_search.fit(X_train_scaled, y_train_encoded)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Train the best model
best_model = grid_search.best_estimator_

# Predict on the test set
y_test_pred = best_model.predict(X_test_scaled)

# Decode the predicted labels
y_test_pred_labels = le.inverse_transform(y_test_pred)

# Create the submission DataFrame
submission = pd.DataFrame({
    'Image': test['Image'],
    'Category': y_test_pred_labels
})



# Save the submission file
submission.to_csv('submission_2_sulay.csv', index=False)
print("Submission file created successfully!")

# prompt: value counts for submision accuracy

# Value counts for submission accuracy
print(submission['Category'].value_counts())


