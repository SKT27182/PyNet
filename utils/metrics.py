import numpy as np
import pandas as pd


class Metrics:

    
    def predict_classes(probability):

        """
        Predicts the class for a given input.

        Parameters:

        - probability (numpy.ndarray):
            A 2D array of shape (m_samples, n_features) representing m_samples, each with n_features.

        Returns:

        - result (numpy.ndarray):
            A list of 1D arrays of shape (m_samples, ) representing the predicted output for each of the m_samples.

        """

        return np.argmax(probability, axis=1)

    def accuracy( y_true, y_pred):

        """
        Calculates the accuracy of the model.

        Parameters:

        - y_true (numpy.ndarray):
            A 1D array of shape (m_samples, ) representing m_samples each with it's label output.

        - y_pred (numpy.ndarray):
            A 1D array of shape (m_samples, ) representing m_samples each with it's label output.

        Returns:

        - result (float):
            The accuracy of the model.

        """

        return np.mean(y_true == y_pred)

    def confusion_matrix( y_true, y_pred):

        """
        Calculates the confusion matrix of the model.

        Parameters:

        - y_true (numpy.ndarray):
            A 1D array of shape (m_samples, ) representing m_samples each with it's label output.

        - y_pred (numpy.ndarray):
            A 1D array of shape (m_samples, ) representing m_samples each with it's label output.

        Returns:

        - result (numpy.ndarray):
            The confusion matrix of the model.

        """

        return pd.crosstab(y_true, y_pred, rownames=["Actual"], colnames=["Predicted"])

    def r2_score( y_true, y_pred):

        """
        Calculates the r2 score of the model.

        Parameters:

        - y_true (numpy.ndarray):
            A 1D array of shape (m_samples, ) representing m_samples each with it's label output.

        - y_pred (numpy.ndarray):
            A 1D array of shape (m_samples, ) representing m_samples each with it's label output.

        Returns:

        - result (float):
            The r2 score of the model.

        """

        return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
 
    def precision(y_true, y_pred, average='macro'):
        '''
        Calculates precision for binary and multiclass classification.
        Parameters
        ----------
        y_true : numpy array, shape (n_samples,)
            Ground truth labels.
        y_pred : numpy array, shape (n_samples,)
            Predicted labels.
        average : str, optional (default='micro')
            Set 'binary' for binary classification and 'micro' for multi-class classification
        Returns
        -------
        precision : float
            Precision value
        '''
        if average not in ['binary', 'macro', 'weighted']:
            raise ValueError("Invalid average type. Must be one of 'binary', 'macro', 'weighted'")

        # If binary classification
        if average == 'binary':
            true_positives = np.sum(y_true * y_pred)
            total_predicted_positives = np.sum(y_pred)
            precision = true_positives/total_predicted_positives
            return precision
        # If multi-class classification
        else:
            if average == 'macro':
                # Calculate precision for each class and then average
                classes = np.unique(y_true)
                n_classes = len(classes)
                precision = 0
                for c in classes:
                    true_positives = np.sum((y_true == c) & (y_pred == c))
                    total_predicted_positives = np.sum(y_pred == c)
                    class_precision = true_positives/total_predicted_positives if total_predicted_positives > 0 else 0
                    precision += class_precision
                return precision/n_classes
            elif average == 'weighted':
                # Calculate precision for each class and then average weighted by support (the number of true instances for each label)
                classes = np.unique(y_true)
                n_classes = len(classes)
                precision = 0
                class_weights = np.array([np.sum(y_true == c) for c in classes])/len(y_true)
                for c, w in zip(classes, class_weights):
                    true_positives = np.sum((y_true == c) & (y_pred == c))
                    total_predicted_positives = np.sum(y_pred == c)
                    class_precision = true_positives/total_predicted_positives if total_predicted_positives > 0 else 0
                    precision += w * class_precision
                return precision

    def recall( y_true, y_pred, average='macro'):

        """
        Calculates the recall of the model.

        Parameters:

        - y_true (numpy.ndarray):
            A 1D array of shape (m_samples, ) representing m_samples each with it's label output.

        - y_pred (numpy.ndarray):
            A 1D array of shape (m_samples, ) representing m_samples each with it's label output.

        Returns:

        - result (float):
            The recall of the model.

        """

        if average not in ['binary', 'macro', 'weighted']:
            raise ValueError("Invalid average type. Must be one of 'binary', 'macro', 'weighted'")

        # If binary classification
        if average == 'binary':
            true_positives = np.sum(y_true * y_pred)
            total_actual_positives = np.sum(y_true)
            recall = true_positives/total_actual_positives
            return recall

        # If multi-class classification
        else:
            if average == 'macro':
                # Calculate recall for each class and then average
                classes = np.unique(y_true)
                n_classes = len(classes)
                recall = 0
                for c in classes:
                    true_positives = np.sum((y_true == c) & (y_pred == c))
                    total_actual_positives = np.sum(y_true == c)
                    class_recall = true_positives/total_actual_positives if total_actual_positives > 0 else 0
                    recall += class_recall
                return recall/n_classes
            elif average == 'weighted':
                # Calculate recall for each class and then average weighted by support (the number of true instances for each label)
                classes = np.unique(y_true)
                n_classes = len(classes)
                recall = 0
                class_weights = np.array([np.sum(y_true == c) for c in classes])/len(y_true)
                for c, w in zip(classes, class_weights):
                    true_positives = np.sum((y_true == c) & (y_pred == c))
                    total_actual_positives = np.sum(y_true == c)
                    class_recall = true_positives/total_actual_positives if total_actual_positives > 0 else 0
                    recall += w * class_recall
                return recall

    def f1_score( y_true, y_pred, average='macro'):
        """
        Calculates the f1 score of the model.

        Parameters:

        - y_true (numpy.ndarray):
            A 1D array of shape (m_samples, ) representing m_samples each with it's label output.

        - y_pred (numpy.ndarray):
            A 1D array of shape (m_samples, ) representing m_samples each with it's label output.

        Returns:

        - result (float):
            The f1 score of the model.

        """

        # calculate the precision and recall  using the methods of this class

        precision = Metrics.precision(y_true, y_pred, average=average)
        recall = Metrics.recall(y_true, y_pred, average=average)

        # calculate the f1 score
        f1 = 2 * (precision * recall) / (precision + recall)

        return f1
