import numpy as np

class Vectorizer:
    """
        Transform raw data into feature vectors. Support ordinal, numerical and categorical data.
        Also implements feature normalization and scaling.

        TODO: Support numerical, ordinal, categorical, histogram features.
    """
    def __init__(self, feature_config, num_bins=5):
        self.feature_config = feature_config
        self.feature_transforms = {}
        self.is_fit = False

    def get_numerical_vectorizer(self, values, verbose=False):
        """
        :return: function to map numerical x to a zero mean, unit std dev normalized score.
        """
        values = np.array(values).astype(float)
        mean, std = np.mean(values), np.std(values)

        # raise NotImplementedError("Numerical vectorizer not implemented yet")

        def vectorizer(x):
            """
            :param x: numerical value
            Return transformed score

            Hint: this fn knows mean and std from the outer scope
            """
            # NotImplementedError("Not implemented")

            # Standard space: z-score = (x - mu)/sigma
            return (float(x) - mean)/std

        return vectorizer

    def get_histogram_vectorizer(self, values):
        raise NotImplementedError("Histogram vectorizer not implemented yet")

    def get_categorical_vectorizer(self, values):
        """
        :return: function to map categorical x to one-hot feature vector
        """
        # raise NotImplementedError("Categorical vectorizer not implemented yet")

        def vectoriser(x):
            pass
        
        return vectoriser

    def fit(self, X):
        """
            Leverage X to initialize all the feature vectorizers (e.g. compute means, std, etc)
            and store them in self.

            This implementation will depend on how you design your feature config.
        """
        # raise NotImplementedError("Not implemented yet")
        features = {}
        vectorisers = {
            "numerical": self.get_numerical_vectorizer,
            # "categorical": self.get_categorical_vectorizer,
        }
        for datapoint in X:
            for feature, value in datapoint.items():
                if feature not in features:   
                    features[feature] = []
                features[feature].append(value)
        
        for feature in features:
            for config in self.feature_config:
                if feature in self.feature_config[config]:
                    self.feature_transforms[feature] = vectorisers[config](features[feature])
                    break

        self.is_fit = True


    def transform(self, X):
        """
        For each data point, apply the feature transforms and concatenate the results into a single feature vector.

        :param X: list of dicts, each dict is a datapoint
        """

        if not self.is_fit:
            raise Exception("Vectorizer not intialized! You must first call fit with a training set" )

        transformed_data = []
        for datapoint in X:
            # transformed_datapoint = {}
            row = []
            for feature, value in datapoint.items():
                # transformed_datapoint[feature] = self.feature_transforms[feature](value)
                for config in self.feature_config:
                    if feature in self.feature_config[config]:
                        row.append(self.feature_transforms[feature](value))
                        break    
            #transformed_data.append(transformed_datapoint)
            transformed_data.append(row)

        return np.array(transformed_data)