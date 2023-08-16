import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from progressbar import ProgressBar


class TNNR:

    def __init__(self,
                 f,
                 n_vars=1,
                 seed=1234,
                 batch_size=16,
                 epochs=10000,
                 l2=0.0,
                 n=1000,
                 val_pct=0.05,
                 test_pct=0.05,
                 learning_rate=1,
                 verbosity=0,
                 neurons=64,
                 inverse_problem=False,
                 mdl_wts_file='mdl_wts',
                 mdl_plot_file=None,
                 show_summary=True,
                 show_loss_plot=True,
                 show_rmse=True,
                 range_data_gen=(-10, 10),
                 noise_range_x=None,
                 noise_range_y=None,
                 noise_normal_x=None,
                 noise_normal_y=None,
                 zero_F_training=False,
                 zero_F_testing=False):

        # set the seed
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # set variables
        self.f = f
        self.n_vars = n_vars
        self.batch_size = batch_size
        self.epochs = epochs
        self.l2 = l2
        self.n = n
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.train_pct = 1 - self.val_pct - self.test_pct
        self.learning_rate = learning_rate
        self.verbosity = verbosity
        self.neurons = neurons
        self.inverse_problem = inverse_problem
        self.mdl_wts_file = mdl_wts_file
        self.mdl_plot_file = mdl_plot_file
        self.show_summary = show_summary
        self.show_loss_plot = show_loss_plot
        self.show_rmse = show_rmse
        self.range_data_gen = range_data_gen
        self.noise_range_x = noise_range_x
        self.noise_range_y = noise_range_y
        self.noise_normal_x = noise_normal_x
        self.noise_normal_y = noise_normal_y
        self.zero_F_training = zero_F_training
        self.zero_F_testing = zero_F_testing

        self.cn_transformer = CenterAndNorm()

        self.model = None
        self.train_history = None
        self.x_train_single = None
        self.y_train_single = None
        self.x_val_single = None
        self.y_val_single = None
        self.x_test_single = None
        self.y_test_single = None
        self.x_pred_test = None
        self.y_pred_test = None
        self.rmse_test = None

        self.generate_data_from_func()
        self.create_model()
        self.train_model()
        if self.inverse_problem:
            self.test_model_inverse_problem()
        else:
            self.test_model()

    def create_model(self):

        x_i = tf.keras.layers.Input(shape=(self.x_train_single.shape[-1],), name='x_i')
        x_j = tf.keras.layers.Input(shape=(self.x_train_single.shape[-1],), name='x_j')

        input_layer = tf.keras.layers.Concatenate()([x_i, x_j])

        merged_layer = tf.keras.layers.Dense(self.neurons,
                                             name='hidden1',
                                             activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(self.l2),
                                             activity_regularizer=tf.keras.regularizers.l2(self.l2))(input_layer)
        merged_layer = tf.keras.layers.Dense(self.neurons,
                                             name='hidden2',
                                             activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(self.l2),
                                             activity_regularizer=tf.keras.regularizers.l2(self.l2))(merged_layer)
        output_layer = tf.keras.layers.Dense(1,
                                             name='output',
                                             kernel_regularizer=tf.keras.regularizers.l2(self.l2),
                                             activity_regularizer=tf.keras.regularizers.l2(self.l2))(merged_layer)

        if self.zero_F_training:
            zero_F_output = tf.keras.layers.Lambda(lambda x: x * 0)(output_layer)
            self.model = tf.keras.models.Model(inputs=[x_i, x_j], outputs=zero_F_output, name='TNNR')
        else:
            self.model = tf.keras.models.Model(inputs=[x_i, x_j], outputs=output_layer, name='TNNR')

        if self.show_summary:
            self.model.summary()

    def train_model(self):

        self.model.compile(loss='mse', metrics=['mse'],
                           optimizer=tf.keras.optimizers.Adadelta(learning_rate=self.learning_rate))

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                                         patience=int(1000 / np.sqrt(self.n) + 1),
                                                         verbose=self.verbosity, min_lr=0)

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3 * int(1000 / np.sqrt(self.n) + 1),
                                                      verbose=self.verbosity)
        mcp_save = tf.keras.callbacks.ModelCheckpoint(self.mdl_wts_file + '.hdf5', save_best_only=True,
                                                      monitor='val_loss', mode='min', verbose=self.verbosity)

        self.perform_fit(reduce_lr, early_stop, mcp_save)

        self.model.load_weights(self.mdl_wts_file + '.hdf5')

        # Plot training & test loss values
        plt.plot(self.train_history.history['loss'], label='Train')
        plt.plot(self.train_history.history['val_loss'], label='Val')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        if self.mdl_plot_file is not None:
            plt.savefig(self.mdl_plot_file + '.png')
        if self.show_loss_plot:
            plt.show()
        else:
            plt.close()

    def perform_fit(self, reduce_lr, early_stop, mcp_save):
        self.train_history = self.model.fit(self.generator_sym(self.x_train_single,
                                                               self.y_train_single,
                                                               self.inverse_problem),
                                            steps_per_epoch=len(self.x_train_single) * 10 / self.batch_size,
                                            epochs=self.epochs,
                                            validation_data=self.generator_sym(self.x_val_single,
                                                                               self.y_val_single,
                                                                               self.inverse_problem),
                                            validation_steps=len(self.x_val_single) * 100 / self.batch_size,
                                            callbacks=[reduce_lr, early_stop, mcp_save], verbose=self.verbosity)

    def generate_data_from_func(self):

        # x_full = np.random.sample([self.n, self.n_vars]) * 2 - 1
        # x_full = np.random.sample([self.n, self.n_vars]) * 40 - 20
        # x_full = np.random.sample([self.n, self.n_vars]) * 20 - 10
        x_full = ((self.range_data_gen[1] - self.range_data_gen[0]) *
                  np.random.sample([self.n, self.n_vars])) + self.range_data_gen[0]
        y_full = np.array([self.f(x) for x in x_full]).flatten()

        if self.noise_range_x is not None:
            x_full = x_full + (
                    (self.noise_range_x[1] - self.noise_range_x[0]) * np.random.sample(x_full.shape) +
                    self.noise_range_x[0])
        if self.noise_range_y is not None:
            y_full = y_full + (
                    (self.noise_range_y[1] - self.noise_range_y[0]) * np.random.sample(y_full.shape) +
                    self.noise_range_y[0])

        if self.noise_normal_x is not None:
            x_full = x_full + np.random.normal(self.noise_normal_x[0], self.noise_normal_x[1], x_full.shape)
        if self.noise_normal_y is not None:
            y_full = y_full + np.random.normal(self.noise_normal_y[0], self.noise_normal_y[1], y_full.shape)

        # split data
        n_total = x_full.shape[0]
        n_val = int(n_total * self.val_pct)
        n_test = int(n_total * self.test_pct)

        all_indices = np.random.permutation(n_total)
        split1 = n_total - n_val - n_test
        split2 = n_total - n_test
        train_indices = all_indices[:split1]
        val_indices = all_indices[split1:split2]
        test_indices = all_indices[split2:]

        x_train_single = x_full[train_indices]
        y_train_single = y_full[train_indices]
        x_val_single = x_full[val_indices]
        y_val_single = y_full[val_indices]
        x_test_single = x_full[test_indices]
        y_test_single = y_full[test_indices]

        self.x_train_single, self.y_train_single = self.cn_transformer.fit_transform(x_train_single, y_train_single)
        self.x_val_single, self.y_val_single = self.cn_transformer.transform(x_val_single, y_val_single)
        self.x_test_single, self.y_test_single = self.cn_transformer.transform(x_test_single, y_test_single)

    def generator_sym(self, x_data, y_data, inverse_problem, num_neighbors=0):

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        batch_size_sym = self.batch_size // 2

        num_single_samples = x_data.shape[0]
        batches_per_sweep = num_single_samples / batch_size_sym
        counter = 0
        counter2 = 0
        perm1 = np.random.permutation(len(x_data))
        perm2 = np.random.permutation(len(x_data))

        while True:

            x1_batch = np.array(x_data[perm1][batch_size_sym * counter:batch_size_sym * (counter + 1)])\
                .astype('float32')
            x2_batch = np.array(x_data[perm2][batch_size_sym * counter:batch_size_sym * (counter + 1)])\
                .astype('float32')
            y1_batch = np.array(y_data[perm1][batch_size_sym * counter:batch_size_sym * (counter + 1)])\
                .astype('float32')
            y2_batch = np.array(y_data[perm2][batch_size_sym * counter:batch_size_sym * (counter + 1)])\
                .astype('float32')

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
                counter2 += 1

                perm2 = np.roll(perm2, 1, axis=0)

            if counter2 >= len(x_data):
                counter2 = 0

                perm1 = np.random.permutation(len(x_data))
                perm2 = np.random.permutation(len(x_data))

    def test_model(self):
        self.y_pred_test = []
        y_pred_r_test = []
        y_pred_check_test = []
        y_median_test = []
        y_var_test = []
        y_mse_test = []

        # for i in ProgressBar()(range(len(self.x_test_single))):
        for i in (range(len(self.x_test_single))):
            pair_b = np.array([self.x_test_single[i]] * len(self.x_train_single))
            diff_a = self.model.predict([pair_b, self.x_train_single], verbose=self.verbosity).flatten()
            diff_b = self.model.predict([self.x_train_single, pair_b], verbose=self.verbosity).flatten()

            if self.zero_F_testing:
                self.y_pred_test.append(self.y_train_single)
            else:
                self.y_pred_test.append(np.average(0.5 * diff_a - 0.5 * diff_b + self.y_train_single, weights=None))

            y_pred_r_test.append(np.average(-diff_b + self.y_train_single))
            y_pred_check_test.append(np.var(0.5 * diff_a + 0.5 * diff_b))
            y_median_test.append(np.median(diff_a + self.y_train_single))
            y_var_test.append(np.var(0.5 * diff_a - 0.5 * diff_b + self.y_train_single))
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
        self.y_pred_test = []
        x_mse_test = []
        y_mse_test = []

        # for i in ProgressBar()(range(len(self.x_test_single))):
        for i in (range(len(self.y_test_single))):
            pair_b = np.array([self.y_test_single[i]] * len(self.y_train_single))
            diff_a = self.model.predict([pair_b, self.y_train_single], verbose=self.verbosity).flatten()
            diff_b = self.model.predict([self.y_train_single, pair_b], verbose=self.verbosity).flatten()

            if self.zero_F_testing:
                self.x_pred_test.append(self.x_train_single)
            else:
                self.x_pred_test.append(np.average(0.5 * diff_a - 0.5 * diff_b + self.x_train_single, weights=None))

            x_mse_test.append((self.x_pred_test[i] - self.x_test_single[i]) ** 2)
            y_mse_test.append((self.f(self.cn_transformer.inverse_transform_x(self.x_pred_test[i])) -
                               self.f(self.cn_transformer.inverse_transform_x(self.x_test_single[i]))) ** 2)

        self.x_pred_test = self.cn_transformer.inverse_transform_x(np.array(self.x_pred_test))
        x_mse_test = np.array(x_mse_test) * self.cn_transformer.x_max ** 2
        y_mse_test = np.average(y_mse_test) ** 0.5
        x_self_check_test = np.abs(np.array(
            self.model.predict(
                [self.y_test_single, self.y_test_single],
                verbose=self.verbosity))).flatten() * self.cn_transformer.x_max

        self.rmse_test = np.average(x_mse_test) ** 0.5
        if self.show_rmse:
            print('Test RMSE (plain):          ', self.rmse_test)
            print('Test RMSE (back to y-space):', np.average(y_mse_test) ** 0.5)


class ANNR(TNNR):
    def create_model(self):

        input_layer = tf.keras.layers.Input(shape=(self.x_train_single.shape[-1],), name='input')

        merged_layer = tf.keras.layers.Dense(self.neurons,
                                             name='hidden1',
                                             activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(self.l2),
                                             activity_regularizer=tf.keras.regularizers.l2(self.l2))(input_layer)
        merged_layer = tf.keras.layers.Dense(self.neurons,
                                             name='hidden2',
                                             activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(self.l2),
                                             activity_regularizer=tf.keras.regularizers.l2(self.l2))(merged_layer)
        output_layer = tf.keras.layers.Dense(1,
                                             name='output',
                                             kernel_regularizer=tf.keras.regularizers.l2(self.l2),
                                             activity_regularizer=tf.keras.regularizers.l2(self.l2))(merged_layer)
        self.model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name='ANNR')

        if self.show_summary:
            self.model.summary()

    def generator_sym(self, x_data, y_data, inverse_problem):

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        batch_size_sym = self.batch_size // 2

        num_single_samples = x_data.shape[0]
        batches_per_sweep = num_single_samples / batch_size_sym
        counter = 0
        counter2 = 0
        perm1 = np.random.permutation(len(x_data))

        while True:

            x_batch = np.array(x_data[perm1][batch_size_sym * counter:batch_size_sym * (counter + 1)])\
                .astype('float32')
            y_batch = np.array(y_data[perm1][batch_size_sym * counter:batch_size_sym * (counter + 1)])\
                .astype('float32')

            counter += 1
            if inverse_problem:
                yield y_batch, x_batch
            else:
                yield x_batch, y_batch

            # restart counter to yield data in the next epoch as well
            if counter >= batches_per_sweep:
                counter = 0
                counter2 += 1

            if counter2 >= len(x_data):
                counter2 = 0
                perm1 = np.random.permutation(len(x_data))

    def test_model(self):

        y_pred = self.model.predict(self.x_test_single, verbose=self.verbosity).flatten()
        self.y_pred_test = self.cn_transformer.inverse_transform_y(y_pred)
        y_mse_test = (self.y_pred_test - self.y_test_single) ** 2
        y_mse_test = np.array(y_mse_test) * self.cn_transformer.y_max ** 2
        self.rmse_test = np.average(y_mse_test) ** 0.5
        if self.show_rmse:
            print('Test RMSE:', self.rmse_test)

    def test_model_inverse_problem(self):

        x_pred = self.model.predict(self.y_test_single, verbose=self.verbosity).flatten()
        self.x_pred_test = self.cn_transformer.inverse_transform_x(x_pred)
        x_mse_test = (self.x_pred_test - self.x_test_single) ** 2
        x_mse_test = np.array(x_mse_test) * self.cn_transformer.x_max ** 2
        self.rmse_test = np.average(x_mse_test) ** 0.5
        if self.show_rmse:
            print('Test RMSE (inverse problem):', self.rmse_test)


class CenterAndNorm:

    def __init__(self):
        self.x_mean = 0
        self.x_max = 0
        self.y_mean = 0
        self.y_max = 0

    def fit(self, x, y):
        self.x_mean = np.mean(x, axis=0)
        self.x_max = np.max(x - self.x_mean, axis=0)

        # self.y_mean = 0
        # self.y_max = 1

        self.y_mean = np.mean(y, axis=0)
        self.y_max = np.max(y - self.y_mean, axis=0)

    def fit_x(self, x):
        self.x_mean = np.mean(x, axis=0)
        self.x_max = np.max(x - self.x_mean, axis=0)

    def transform_x(self, x):
        x_new = x.copy()

        x_new -= self.x_mean
        x_new /= self.x_max

        return x_new

    def transform_y(self, y):
        y_new = y.copy()

        y_new -= self.y_mean
        y_new /= self.y_max

        return y_new

    def transform(self, x, y):
        return self.transform_x(x), self.transform_y(y)

    def inverse_transform_x(self, x):
        x_new = x.copy()

        x_new *= self.x_max
        x_new += self.x_mean

        return x_new

    def inverse_transform_y(self, y):
        y_new = y.copy()

        y_new *= self.y_max
        y_new += self.y_mean

        return y_new

    def inverse_transform(self, x, y):
        return self.inverse_transform_x(x), self.inverse_transform_y(y)

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x, y)

    def fit_transform_x(self, x):
        self.fit_x(x)
        return self.transform_x(x)
