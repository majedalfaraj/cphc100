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
        values = np.array([v for v in values if not Vectorizer.is_missing(v)]).astype(float)
        mean, std = np.mean(values), np.std(values)

        # raise NotImplementedError("Numerical vectorizer not implemented yet")

        def vectorizer(x):
            """
            :param x: numerical value
            Return transformed score

            Hint: this fn knows mean and std from the outer scope
            """
            # Standard space: z-score = (x - mu)/sigma

            if Vectorizer.is_missing(x):
                x = mean

            return (float(x) - mean)/std

        return vectorizer

    def get_histogram_vectorizer(self, values):
        # raise NotImplementedError("Histogram vectorizer not implemented yet")
        num_bins = 5
        lower_bound = min(values)
        upper_bound = max(values)
        bin_width = (upper_bound - lower_bound)/num_bins

        def vectoriser(x):
            if Vectorizer.is_missing(x):
                x = np.mean(values)

            ans = np.zeros(num_bins)
            location = min(int((x - lower_bound) // bin_width), num_bins - 1)
            ans[location] = 1
            return ans

        return vectoriser

    def get_categorical_vectorizer(self, values):
        """
        :return: function to map categorical x to one-hot feature vector
        """
        values = np.unique(np.array(values)).astype(str)
        def vectoriser(x):
            if Vectorizer.is_missing(x):
                return np.zeros(len(values))
            return (values == x).astype(int)
        
        return vectoriser
    
    def get_ordinal_vectoriser(self, values):
        """
        :return: function to map ordinal x to numerical value from [0, n-1]
        """
        seen = set()
        values = [x for x in values if x not in seen and not seen.add(x)]
        def vectoriser(x):
            if Vectorizer.is_missing(x) or x not in values:
                return 0
            return values.index(x)

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
            "categorical": self.get_categorical_vectorizer,
            "ordinal": self.get_ordinal_vectoriser,
            "histogram": self.get_histogram_vectorizer,
        }
        for datapoint in X:
            for feature, value in datapoint.items():
                if feature not in features:   
                    features[feature] = []
                features[feature].append(value)

        self.feature_vectorisers = {}
        for config, feats in self.feature_config.items():
            for feat in feats:
                self.feature_vectorisers[feat] = config
        
        for feature in features:
            if feature not in self.feature_vectorisers:
                continue
            config = self.feature_vectorisers[feature]
            self.feature_transforms[feature] = vectorisers[config](features[feature])

        self.features_ordered = list(self.feature_transforms.keys())
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
            for feature in self.features_ordered:
                if feature not in datapoint:
                    continue
                value = datapoint[feature]
                # transformed_datapoint[feature] = self.feature_transforms[feature](value)
                transformed = self.feature_transforms[feature](value)
                row.extend(transformed if isinstance(transformed, (list, np.ndarray)) else [transformed])
            #transformed_data.append(transformed_datapoint)
            transformed_data.append(row)

        return np.array(transformed_data)
    

    def is_missing(val):
        return val == "" or val is None or (isinstance(val, float) and np.isnan(val)) or (isinstance(val, str) and val.strip() in [".", ".F", ".M", ".A", ".N", ".NA", "NA", "N/A", ""]) 