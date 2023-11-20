import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Loaded data.
    """
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def calculate_thresholds(df: pd.DataFrame, column: str, num_levels: int) -> list:
    """
    Calculate thresholds for engagement levels.

    Args:
    df (pd.DataFrame): Data.
    column (str): Column name for which to calculate thresholds.
    num_levels (int): Number of engagement levels.

    Returns:
    list: Threshold values.
    """
    min_rate = df[column].min()
    max_rate = df[column].max()
    range_eighth = (max_rate - min_rate) / num_levels
    return [min_rate + range_eighth * i for i in range(1, num_levels+1)]

def assign_engagement_level(rate: float, thresholds: list) -> str:
    """
    Assign engagement level based on rate.

    Args:
    rate (float): Engagement rate.
    thresholds (list): Threshold values for engagement levels.

    Returns:
    str: Engagement level.
    """
    for i, threshold in enumerate(thresholds):
        if rate <= threshold:
            return f'Level {i+1}'
    return f'Level {len(thresholds)+1}'

def encode_data(df: pd.DataFrame, column: str) -> list:
    """
    Encode text data into embeddings.

    Args:
    df (pd.DataFrame): Data.
    column (str): Column name of text to encode.

    Returns:
    list: Encoded data.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2', device = "cuda")
    return model.encode(df[column], show_progress_bar = True)

def encode_labels(df: pd.DataFrame, column: str) -> tuple:
    """
    Encode labels into integers and return the encoder.

    Args:
    df (pd.DataFrame): Data.
    column (str): Column name of labels to encode.

    Returns:
    tuple: Encoded labels and the LabelEncoder instance.
    """
    encoder = LabelEncoder()
    encoder.fit(df[column])
    return encoder.transform(df[column]), encoder


def split_data(X: list, y: list, test_size: float, random_state: int):
    """
    Split data into training and test sets.

    Args:
    X (list): Features.
    y (list): Labels.
    test_size (float): Proportion of data to include in the test split.
    random_state (int): Random state.

    Returns:
    tuple: Training and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
def prepare_data(file_path: str, engagement_rate_col: str, text_col: str, engagement_level_col: str, num_levels: int, test_size: float, random_state: int) -> tuple:
    """
    Load and prepare data for machine learning.

    Args:
    file_path (str): Path to the CSV file.
    engagement_rate_col (str): Column name for engagement rate.
    text_col (str): Column name for text data.
    engagement_level_col (str): Column name for engagement level.
    num_levels (int): Number of engagement levels.
    test_size (float): Proportion of data to include in the test split.
    random_state (int): Random state.

    Returns:
    tuple: Training and test sets, and the LabelEncoder instance.
    """
    logger.info("Preparing data")

    # Load data
    df = load_data(file_path)

    # Calculate thresholds
    thresholds = calculate_thresholds(df, engagement_rate_col, num_levels)

    # Assign engagement levels
    df[engagement_level_col] = df[engagement_rate_col].apply(lambda x: assign_engagement_level(x, thresholds))

    # Shuffle data
    df = df.sample(frac=1).reset_index(drop=True)

    # Encode data and labels
    X_array = encode_data(df, text_col)
    y_array, encoder = encode_labels(df, engagement_level_col)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = split_data(X_array, y_array, test_size, random_state)
    
    logger.info("Data preparation complete")
    return (X_train, X_test, y_train, y_test), encoder

