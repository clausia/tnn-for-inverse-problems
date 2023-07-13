from TwinNeuralNetwork import TNNR
from sklearn.neighbors import NearestNeighbors
import numpy as np


class TNNR_NN(TNNR):

    def __init__(self, *args, num_nn_train=2, num_nn_test=1, radius_nn_predict=1, **kwargs):
        self.num_nn_train = num_nn_train
        self.num_nn_test = num_nn_test
        self.radius_nn_predict = radius_nn_predict
        super(TNNR_NN, self).__init__(*args, **kwargs)

    def perform_fit(self, reduce_lr, early_stop, mcp_save):
        self.train_history = self.model.fit(self.generator_sym(self.x_train_single,
                                                               self.y_train_single,
                                                               self.inverse_problem,
                                                               num_neighbors=self.num_nn_train),
                                            steps_per_epoch=len(self.x_train_single) * 10 / self.batch_size,
                                            epochs=self.epochs,
                                            validation_data=self.generator_sym(self.x_val_single,
                                                                               self.y_val_single,
                                                                               self.inverse_problem,
                                                                               num_neighbors=
                                                                               int(self.num_nn_train * self.val_pct /
                                                                                   self.train_pct)),
                                            validation_steps=len(self.x_val_single) * 100 / self.batch_size,
                                            callbacks=[reduce_lr, early_stop, mcp_save], verbose=self.verbosity)

    @staticmethod
    def get_nearest_neighbors(x_new, x_reference, num_neighbors):
        neighbor_finder = NearestNeighbors(n_neighbors=num_neighbors)
        neighbor_finder.fit(x_reference)
        return neighbor_finder.kneighbors(x_new, return_distance=False)

    @staticmethod
    def get_nearest_neighbors_radius(x_new, x_reference, radius_neighbors):
        neighbor_finder = NearestNeighbors(radius=radius_neighbors)
        neighbor_finder.fit(x_reference)
        return neighbor_finder.radius_neighbors(x_new, return_distance=False)

    def generator_sym(self, x_data, y_data, inverse_problem, num_neighbors=0):

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        num_neighbors = np.max([2, int(num_neighbors)])
        nn_indexes = self.get_nearest_neighbors(x_data, x_data, num_neighbors)
        [np.random.shuffle(x) for x in nn_indexes]

        batch_size_sym = self.batch_size // 2

        num_single_samples = x_data.shape[0]
        batches_per_sweep = num_single_samples / batch_size_sym
        counter = 0

        perm1 = np.random.permutation(len(x_data))

        i1, = np.random.choice(range(num_neighbors), 1, replace=False)

        x1 = x_data[perm1]
        x2 = x_data[nn_indexes[perm1][:, i1]]

        y1 = y_data[perm1]
        y2 = y_data[nn_indexes[perm1][:, i1]]

        while True:

            x1_batch = np.array(x1[batch_size_sym * counter:batch_size_sym * (counter + 1)]).astype('float32')
            x2_batch = np.array(x2[batch_size_sym * counter:batch_size_sym * (counter + 1)]).astype('float32')
            y1_batch = np.array(y1[batch_size_sym * counter:batch_size_sym * (counter + 1)]).astype('float32')
            y2_batch = np.array(y2[batch_size_sym * counter:batch_size_sym * (counter + 1)]).astype('float32')

            x_a = np.concatenate((x1_batch, x2_batch), axis=0)
            x_b = np.concatenate((x2_batch, x1_batch), axis=0)

            y_a = np.concatenate((y1_batch, y2_batch), axis=0)
            y_b = np.concatenate((y2_batch, y1_batch), axis=0)

            counter += 1
            if inverse_problem:
                yield [y_a, y_b], x_a - x_b
            else:
                yield [x_a, x_b], y_a - y_b

            # restart counter to yield data in the next epoch as well
            if counter >= batches_per_sweep:
                counter = 0

                perm1 = np.random.permutation(len(x_data))

                i1, = np.random.choice(range(num_neighbors), 1, replace=False)

                [np.random.shuffle(x) for x in nn_indexes]

                x1 = x_data[perm1]
                x2 = x_data[nn_indexes[perm1][:, i1]]

                y1 = y_data[perm1]
                y2 = y_data[nn_indexes[perm1][:, i1]]

    def test_model(self):
        self.y_pred_test = []
        y_pred_r_test = []
        y_pred_check_test = []
        y_median_test = []
        y_var_test = []
        y_mse_test = []

        nn_indexes = self.get_nearest_neighbors(self.x_test_single, self.x_train_single, self.num_nn_test)

        for i in (range(len(self.x_test_single))):

            pair_b = np.array([self.x_test_single[i]] * self.num_nn_test)
            diff_a = self.model.predict([pair_b, self.x_train_single[nn_indexes[i]]], verbose=self.verbosity).flatten()
            diff_b = self.model.predict([self.x_train_single[nn_indexes[i]], pair_b], verbose=self.verbosity).flatten()

            if self.zero_F_testing:
                self.y_pred_test.append(self.y_train_single[nn_indexes[i]])
            else:
                self.y_pred_test.append(np.average(0.5 * diff_a - 0.5 * diff_b + self.y_train_single[nn_indexes[i]],
                                                   weights=None))

            y_pred_r_test.append(np.average(-diff_b + self.y_train_single[nn_indexes[i]]))
            y_pred_check_test.append(np.var(0.5 * diff_a + 0.5 * diff_b))
            y_median_test.append(np.median(diff_a + self.y_train_single[nn_indexes[i]]))
            y_var_test.append(np.var(0.5 * diff_a - 0.5 * diff_b + self.y_train_single[nn_indexes[i]]))
            y_mse_test.append((self.y_pred_test[i] - self.y_test_single[i]) ** 2)

        self.y_pred_test = self.cn_transformer.inverse_transform_y(np.array(self.y_pred_test))
        y_pred_r_test = self.cn_transformer.inverse_transform_y(np.array(y_pred_r_test))
        y_pred_check_test = np.array(y_pred_check_test) * self.cn_transformer.y_max
        y_median_test = self.cn_transformer.inverse_transform_y(np.array(y_median_test))
        y_var_test = np.array(y_var_test) * self.cn_transformer.y_max ** 2
        y_mse_test = np.array(y_mse_test) * self.cn_transformer.y_max ** 2
        y_self_check_test = np.abs(np.array(
            self.model.predict(
                [self.x_test_single, self.x_test_single],
                verbose=self.verbosity))).flatten() * self.cn_transformer.y_max

        self.rmse_test = np.average(y_mse_test) ** 0.5
        if self.show_rmse:
            print('Test RMSE:', self.rmse_test)

    def test_model_inverse_problem(self):
        self.x_pred_test = []
        x_mse_test = []

        nn_indexes = self.get_nearest_neighbors(self.y_test_single.reshape(-1, 1), self.y_train_single.reshape(-1, 1),
                                                self.num_nn_test)

        for i in (range(len(self.y_test_single))):
            pair_b = np.array([self.y_test_single[i]] * self.num_nn_test)
            diff_a = self.model.predict([pair_b, self.y_train_single[nn_indexes[i]]], verbose=self.verbosity).flatten()
            diff_b = self.model.predict([self.y_train_single[nn_indexes[i]], pair_b], verbose=self.verbosity).flatten()

            if self.zero_F_testing:
                self.x_pred_test.append(self.x_train_single[nn_indexes[i]])
            else:
                self.x_pred_test.append(np.average(0.5 * diff_a - 0.5 * diff_b + self.x_train_single[nn_indexes[i]],
                                                   weights=None))

            x_mse_test.append((self.x_pred_test[i] - self.x_test_single[i]) ** 2)

        self.x_pred_test = self.cn_transformer.inverse_transform_x(np.array(self.x_pred_test))
        x_mse_test = np.array(x_mse_test) * self.cn_transformer.x_max ** 2
        x_self_check_test = np.abs(np.array(
            self.model.predict(
                [self.y_test_single, self.y_test_single],
                verbose=self.verbosity))).flatten() * self.cn_transformer.x_max

        self.rmse_test = np.average(x_mse_test) ** 0.5
        if self.show_rmse:
            print('Test RMSE:', self.rmse_test)

