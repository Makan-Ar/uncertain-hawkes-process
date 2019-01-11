import numpy as np
import matplotlib.pyplot as plt


class HighSchoolData:
    network_data_path = "data/HighSchoolContactFriendships2013/High-School_data_2013.csv"
    preprocessed_network_data_path = "data/HighSchoolContactFriendships2013/preprocessed-High-School_data_2013.txt"
    diaries_data_path = 'data/HighSchoolContactFriendships2013/Contact-diaries-network_data_2013.csv'
    daily_time_intervals = [(1385982020, 1385999980), (1386054020, 1386086380), (1386140420, 1386172780),
                            (1386226820, 1386259180), (1386313220, 1386345580)]
    start_time = 1385982020
    end_time = 1386345580
    interaction_duration = 20
    network_data = None  # raw high school data
    preprocessed_network_data = None  # student1_id, student2_id, interaction_start_time, duration, day number
    diaries_data_self_reported = None  # diary data keyed by the person who reported
    diaries_data_peer_reported = None  # diary data keyed by the person who was reported
    __student_ids = None

    def __init__(self, preprocessed=True):
        self.network_data = np.genfromtxt(self.network_data_path, dtype=int, usecols=(0, 1, 2))
        self.get_diaries_data()
        if preprocessed:
            self.preprocessed_network_data = np.genfromtxt(self.preprocessed_network_data_path, dtype=int)

    def get_diaries_data(self):
        raw_diaries_data = np.genfromtxt(self.diaries_data_path, dtype=int)
        self.diaries_data_self_reported = {}
        self.diaries_data_peer_reported = {}
        for i in range(np.shape(raw_diaries_data)[0]):
            if raw_diaries_data[i, 0] not in self.diaries_data_self_reported:
                self.diaries_data_self_reported[raw_diaries_data[i, 0]] = {}
            self.diaries_data_self_reported[raw_diaries_data[i, 0]][raw_diaries_data[i, 1]] = raw_diaries_data[i, 2]

            if raw_diaries_data[i, 1] not in self.diaries_data_peer_reported:
                self.diaries_data_peer_reported[raw_diaries_data[i, 1]] = {}
            self.diaries_data_peer_reported[raw_diaries_data[i, 1]][raw_diaries_data[i, 0]] = raw_diaries_data[i, 2]

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

        self.preprocessed_network_data = np.zeros((total_interactions, 5), dtype=int)
        cnt = 0
        day_number = 0
        for student_one, student_two in temp_inters:
            for i in range(len(temp_inters[(student_one, student_two)])):
                s_time = temp_inters[(student_one, student_two)][i][0]
                for enu, t in enumerate(self.daily_time_intervals):
                    if (t[0] - self.start_time) <= s_time <= (t[1] - self.start_time):
                        day_number = enu + 1
                        break

                self.preprocessed_network_data[cnt] = [student_one, student_two, s_time,
                                                       temp_inters[(student_one, student_two)][i][1], day_number]
                cnt += 1

        self.preprocessed_network_data = self.preprocessed_network_data[self.preprocessed_network_data[:, 2].argsort()]

        if save_as_text is not None:
            np.savetxt(save_as_text, self.preprocessed_network_data, fmt='%d', delimiter=' ')

        return self.preprocessed_network_data

    def get_student_preprocessed_interactions(self, student_id):
        if self.preprocessed_network_data is None:
            self.run_preprocessing()
        return self.preprocessed_network_data[np.where(self.preprocessed_network_data[:, 0:2] == student_id)[0]]

    def plot_student_interactions(self, student_id=None, based_on_self_reported_diary_only=False, separate_days=True):
        """
        Plots student interactions as a scatter plot with Time on X axis and duration of each interaction on the Y axis.
        :param student_id: ID of the student. If None, all interactions will be plotted.
        :param based_on_self_reported_diary_only: If True, only self reported interactions by the student_id will be
                                                  considered as valid. Will be ignored if `student_id` is None.
        :param separate_days: If True, each day of the data collection will have its own subplot.
        """
        fig_title = "All Student Interactions"
        if student_id is not None:
            interactions = self.get_student_preprocessed_interactions(student_id)
            fig_title = "Student ID: {}".format(student_id)
        else:
            interactions = self.preprocessed_network_data

        is_validated = self.is_interaction_in_diary(interactions, based_on_self_reported_diary_only, student_id)
        validated_interactions = interactions[np.where(is_validated == True)[0], :]
        proximity_interactions = interactions[np.where(is_validated == False)[0], :]

        if separate_days:
            fig, axs = plt.subplots(len(self.daily_time_intervals), 1)
            for i, ax in enumerate(axs):
                d_ind = np.where(proximity_interactions[:, 4] == i + 1)
                p1 = ax.scatter(proximity_interactions[d_ind, 2], proximity_interactions[d_ind, 3], c='red', alpha=0.6)

                d_ind = np.where(validated_interactions[:, 4] == i + 1)
                p2 = ax.scatter(validated_interactions[d_ind, 2], validated_interactions[d_ind, 3], c='blue', alpha=0.6)

                ax.set_title("Day {}".format(i + 1))

            fig.legend((p1, p2), ('Proximity Only Interactions', 'Validated Interactions'), 'upper right')
            fig.text(0.5, 0.03, 'Time (s)', ha='center')
            fig.text(0.03, 0.5, 'Length of Interaction (s)', va='center', rotation='vertical')
            fig.suptitle(fig_title, fontsize=16)
        else:
            plt.scatter(proximity_interactions[:, 2], proximity_interactions[:, 3], c='red', alpha=0.6,
                        label="Proximity Only Interactions")
            plt.scatter(validated_interactions[:, 2], validated_interactions[:, 3], c='blue', alpha=0.6,
                        label="Validated Interactions")
            plt.xlabel("Time (s)")
            plt.ylabel("Length of Interaction (s)")
            plt.title(fig_title)
            plt.legend()

        plt.tight_layout()
        plt.show()

    def is_interaction_in_diary(self, interactions, based_on_self_reported_diary_only=False, student_id=None):
        """
        Check if the interaction is in the diary.
        :param interactions: must be a list in the same format as preprocessed_contact_data or a single row
        :param based_on_self_reported_diary_only: if true, only looks for the interaction in self reported
        :param student_id: must be passed if `based_on_self_reported_diary_only` is True
        :return: boolean or a list of boolean
        """
        if student_id is None and based_on_self_reported_diary_only:
            based_on_self_reported_diary_only = False
            print("Warning: Cannot check for self reported diary without student id")

        return_list = True
        valid_interactions = []
        if len(np.shape(interactions)) == 1:
            interactions = [interactions]
            return_list = False

        for i in range(np.shape(interactions)[0]):
            if based_on_self_reported_diary_only:
                sid = interactions[i, 0] if interactions[i, 0] != student_id else interactions[i, 1]
                valid_interactions.append(sid in self.diaries_data_self_reported[student_id])
            else:
                is_valid = (interactions[i, 0] in self.diaries_data_self_reported and
                            interactions[i, 1] in self.diaries_data_self_reported[interactions[i, 0]]) or \
                            (interactions[i, 1] in self.diaries_data_self_reported and
                             interactions[i, 0] in self.diaries_data_self_reported[interactions[i, 1]])
                valid_interactions.append(is_valid)

        if not return_list:
            return valid_interactions[0]
        return np.array(valid_interactions)


# class BaseLineClassifier(HighSchoolData):
#     def __inti__(self):
#         HighSchoolData.__init__()
        

if __name__ == '__main__':
    high_school_data = HighSchoolData()
    # print(np.shape(high_school_data.preprocessed_contact_data))
    # print(high_school_data.preprocessed_contact_data)
    # for i in range(len(high_school_data.preprocessed_contact_data)):
    #     print(high_school_data.preprocessed_contact_data[i, 2])

    # print(high_school_data.preprocessed_contact_data[:, 2:4])
    # print(np.shape(high_school_data.preprocessed_contact_data))
    # s1 = high_school_data.get_student_preprocessed_interactions(1)
    # s1 = high_school_data.get_student_raw_interactions(1)
    # print(s1[s1[:, 2].argsort()])

    for sid in list(high_school_data.get_student_ids())[:10]:
        high_school_data.plot_student_interactions(sid, separate_days=True)
    
    # high_school_data.plot_student_interactions(separate_days=True)
