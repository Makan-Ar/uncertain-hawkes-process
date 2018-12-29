import numpy as np


class HighSchoolData:
    network_data_path = "data/HighSchoolContactFriendships2013/High-School_data_2013.csv"
    preprocessed_network_data_path = "data/HighSchoolContactFriendships2013/preprocessed-High-School_data_2013.txt"
    start_time = 1385982020
    end_time = 1386345580
    interaction_duration = 20
    network_data = None
    preprocessed_network_data = None
    __student_ids = None

    def __init__(self, preprocessed=False):
        self.network_data = np.genfromtxt(self.network_data_path, dtype=int, usecols=(0, 1, 2))
        if preprocessed:
            self.preprocessed_network_data = np.genfromtxt(self.preprocessed_network_data_path, dtype=int)

    def get_student_ids(self):
        if self.__student_ids is None:
            self.__student_ids = set()
            self.__student_ids = self.__student_ids.union(self.network_data[:, 1])
            self.__student_ids = self.__student_ids.union(self.network_data[:, 2])

        return self.__student_ids

    def get_student_raw_interactions(self, student_id, reset_start_time=False):
        raw_inters = self.network_data[np.where(self.network_data[:, 1:3] == student_id)[0]]
        if reset_start_time:
            raw_inters[:, 0] -= self.start_time
        return raw_inters

    def run_preprocessing(self, save_as_text=None):
        if self.preprocessed_network_data is not None:
            return self.preprocessed_network_data

        # Set starting time to 0
        temp_network_data = self.network_data.copy()
        temp_network_data[:, 0] -= self.start_time

        temp_inters = {}
        total_interactions = 0
        for i in range(np.shape(temp_network_data)[0]):
            inter_key = (temp_network_data[i, 1], temp_network_data[i, 2])

            if (temp_network_data[i, 2], temp_network_data[i, 1]) in temp_inters:
                inter_key = (temp_network_data[i, 2], temp_network_data[i, 1])
            elif inter_key not in temp_inters:
                temp_inters[inter_key] = []

            for j in range(len(temp_inters[inter_key]) - 1, -1, -1):
                if temp_network_data[i, 0] == temp_inters[inter_key][j][0] + temp_inters[inter_key][j][1]:
                    temp_inters[inter_key][j][1] += self.interaction_duration
                    break
            else:
                temp_inters[inter_key].append([temp_network_data[i, 0], self.interaction_duration])
                total_interactions += 1

        self.preprocessed_network_data = np.zeros((total_interactions, 4), dtype=int)
        cnt = 0
        for student_one, student_two in temp_inters:
            for i in range(len(temp_inters[(student_one, student_two)])):
                self.preprocessed_network_data[cnt] = [student_one, student_two,
                                                       temp_inters[(student_one, student_two)][i][0],
                                                       temp_inters[(student_one, student_two)][i][1]]
                cnt += 1

        self.preprocessed_network_data = self.preprocessed_network_data[self.preprocessed_network_data[:, 2].argsort()]

        if save_as_text is not None:
            np.savetxt(save_as_text, self.preprocessed_network_data, fmt='%d', delimiter=' ')

        return self.preprocessed_network_data

    def get_student_preprocessed_interactions(self, student_id):
        if self.preprocessed_network_data is None:
            self.run_preprocessing()
        return self.preprocessed_network_data[np.where(self.preprocessed_network_data[:, 0:2] == student_id)[0]]


if __name__ == '__main__':
    high_school_data = HighSchoolData(preprocessed=True)
    student = high_school_data.get_student_raw_interactions(1, reset_start_time=True)
    # high_school_data.get_student_preprocessed_interactions(1)
    # high_school_data.preprocess(save_as_text="data/HighSchoolContactFriendships2013/preprocessed-High-School_data_2013.txt")
    print(len(high_school_data.get_student_preprocessed_interactions(1)))
