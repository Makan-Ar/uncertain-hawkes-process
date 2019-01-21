import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# contact_data_path = "data/Hospital10/detailed_list_of_contacts_Hospital.dat_"
# co_location_data_path = "data/Hospital10/tij_pres_LH10.dat"
# contact_data_path = "data/SFHH/tij_SFHH.dat_"
# co_location_data_path = "data/SFHH/tij_pres_SFHH.dat"
#
# contact_data = np.genfromtxt(contact_data_path, dtype=int, usecols=(0, 1, 2))
# co_location_data = np.genfromtxt(co_location_data_path, dtype=int)
# co_location_data_dic = {}
#
# for i in range(len(co_location_data)):
#     if co_location_data[i, 0] not in co_location_data_dic:
#         co_location_data_dic[co_location_data[i, 0]] = []
#
#     if co_location_data[i, 1] < co_location_data[i, 2]:
#         co_location_data_dic[co_location_data[i, 0]].append((co_location_data[i, 1], co_location_data[i, 2]))
#     else:
#         co_location_data_dic[co_location_data[i, 0]].append((co_location_data[i, 2], co_location_data[i, 1]))
#
# exists = []
# for i in range(len(contact_data)):
#     if contact_data[i, 0] in co_location_data_dic:
#         temp = (contact_data[i, 1], contact_data[i, 2]) if contact_data[i, 1] < contact_data[i, 2] else \
#             (contact_data[i, 2], contact_data[i, 1])
#
#         if temp in co_location_data_dic[contact_data[i, 0]]:
#             exists.append(contact_data[i, :])

# print(exists)
# print(len(exists))

# 70261 total contact
# 1417485 total co-location
# 36858 matches between both


class CoLocationData:
    dataset_info = {
        "SFHH": {"name": "SFHH"},
        "high_school": {"name": "Thiers13"},
        "hospital": {"name": "LH10"},
        "primary_school": {"name": "LyonSchool"},
        "workplace_13": {"name": "InVS13"},
        "workplace_15": {"name": "InVS15"}
    }
    # base_data_path = "data/co-presence/"
    # base_preprocessed_data_path = "data/co-presence/preprocessed/"
    base_data_path = "/shared/DataSets/SocioPatterns/co-presence/"
    base_preprocessed_data_path = "/shared/Results/HawkesUncertainEvents/datasets/co-presence/preprocessed/"
    dataset_name = None
    day_intervals = None
    interaction_duration = 20
    contact_data = None  # raw contact data
    co_location_data = None  # raw co-location data
    preprocessed_contact_data = {}  # (volunteer1_id, (<) volunteer2_id): [interaction_start_time, duration] for contact
    preprocessed_co_location_data = {}  # (volunteer1_id, volunteer2_id): [interaction_start_time, duration] for co-loc
    preprocessed_co_location_data_list = []  # vol1_id, vol2_id, inter_start_time, duration, is_in_contact, day_number
    contact_data_interactions = {}  # dict, keyed by volunteer id and a list of all people who he/she has interacted w/
    co_location_data_interactions = {}  # same as contact_data_interactions, but for co-location data
    __volunteer_ids = None

    def __init__(self, dataset_name, load_form_pickle=True):
        self.dataset_name = dataset_name
        if not load_form_pickle:
            print("Loading contact data...")
            self.contact_data = np.genfromtxt(self.base_data_path + "contact/tij_" +
                                              self.dataset_info[self.dataset_name]["name"] + ".dat", dtype=int)
            print("Loading co-location data...")
            self.co_location_data = np.genfromtxt(self.base_data_path + "co-presence/tij_pres_" +
                                                  self.dataset_info[self.dataset_name]["name"] + ".dat", dtype=int)
            print("Running preprocessing on contact data...")
            self.run_preprocessing_on_contact_data()
            print("Running preprocessing on co-location data...")
            self.run_preprocessing_on_co_location_data()
            print("Getting day intervals...")
            self.get_day_intervals()
            print("Checking for co-location interactions in contact data...")
            self.__check_for_co_location_interactions_in_contact()

            if not os.path.exists(self.base_preprocessed_data_path):
                os.makedirs(self.base_preprocessed_data_path)

            print("Saving preprocessed data...")
            np.save(self.base_preprocessed_data_path + self.dataset_info[self.dataset_name]["name"],
                    [self.contact_data, self.co_location_data, self.preprocessed_contact_data,
                     self.preprocessed_co_location_data, self.preprocessed_co_location_data_list,
                     self.contact_data_interactions, self.co_location_data_interactions, self.day_intervals])
        else:
            print("Loading preprocessed data...")
            self.contact_data, self.co_location_data, self.preprocessed_contact_data, \
             self.preprocessed_co_location_data, self.preprocessed_co_location_data_list, \
             self.contact_data_interactions, self.co_location_data_interactions, self.day_intervals = \
             np.load(self.base_preprocessed_data_path + self.dataset_info[self.dataset_name]["name"] + ".npy")

        print(f"Initialization Done for {self.dataset_name}!")

    def get_day_intervals(self):
        if self.dataset_name == "hospital":
            self.day_intervals = [[39600, 298780]]
            return self.day_intervals

        day_t_diff = 18000  # number of seconds in 5 hours
        self.day_intervals = []
        day_start_time = self.contact_data[0, 0]
        for i in range(len(self.contact_data) - 1):
            if self.contact_data[i, 0] + day_t_diff <= self.contact_data[i + 1, 0]:
                self.day_intervals.append([day_start_time, self.contact_data[i, 0]])
                day_start_time = self.contact_data[i + 1, 0]
        self.day_intervals.append([day_start_time, self.contact_data[-1, 0]])

        day_number = 0
        day_start_time = self.co_location_data[0, 0]
        for i in range(len(self.co_location_data) - 1):
            if self.co_location_data[i, 0] + day_t_diff <= self.co_location_data[i + 1, 0]:
                self.day_intervals[day_number][0] = min(self.day_intervals[day_number][0], day_start_time)
                self.day_intervals[day_number][1] = max(self.day_intervals[day_number][1], self.co_location_data[i, 0])
                day_start_time = self.co_location_data[i + 1, 0]
                day_number += 1

        self.day_intervals[day_number][0] = min(self.day_intervals[day_number][0], day_start_time)
        self.day_intervals[day_number][1] = max(self.day_intervals[day_number][1], self.co_location_data[-1, 0])
        print(self.day_intervals)
        return self.day_intervals

    def run_preprocessing_on_contact_data(self, save_as_text=None):
        if len(self.preprocessed_contact_data) > 0:
            return self.preprocessed_contact_data

        return self.__run_preprocessing(self.contact_data, self.preprocessed_contact_data,
                                        self.contact_data_interactions, save_as_text)

    def run_preprocessing_on_co_location_data(self, save_as_text=None):
        if len(self.preprocessed_co_location_data) > 0:
            return self.preprocessed_co_location_data

        return self.__run_preprocessing(self.co_location_data, self.preprocessed_co_location_data,
                                        self.co_location_data_interactions, save_as_text)

    def get_volunteer_ids(self):
        if self.__volunteer_ids is None:
            self.__volunteer_ids = set()
            self.__volunteer_ids = self.__volunteer_ids.union(self.contact_data[:, 1])
            self.__volunteer_ids = self.__volunteer_ids.union(self.contact_data[:, 2])

        return self.__volunteer_ids

    def __run_preprocessing(self, raw_data, preprocessed_data, interaction_data, save_as_text=None):
        total_interactions = 0
        for i in range(np.shape(raw_data)[0]):
            inter_key = (raw_data[i, 1], raw_data[i, 2]) if raw_data[i, 1] < raw_data[i, 2] else \
                (raw_data[i, 2], raw_data[i, 1])

            if inter_key not in preprocessed_data:
                preprocessed_data[inter_key] = []
                if inter_key[0] not in interaction_data:
                    interaction_data[inter_key[0]] = []
                interaction_data[inter_key[0]].append(inter_key[1])

                if inter_key[1] not in interaction_data:
                    interaction_data[inter_key[1]] = []
                interaction_data[inter_key[1]].append(inter_key[0])

            for j in range(len(preprocessed_data[inter_key]) - 1, -1, -1):
                if raw_data[i, 0] == preprocessed_data[inter_key][j][0] + preprocessed_data[inter_key][j][1]:
                    preprocessed_data[inter_key][j][1] += self.interaction_duration
                    break
            else:
                preprocessed_data[inter_key].append([raw_data[i, 0], self.interaction_duration])
                total_interactions += 1

        if save_as_text is None:
            return preprocessed_data

        temp_preprocessed_data = np.zeros((total_interactions, 5), dtype=int)

        cnt = 0
        day_number = 0
        for volunteer_one, volunteer_two in preprocessed_data:
            for i in range(len(preprocessed_data[(volunteer_one, volunteer_two)])):
                s_time = preprocessed_data[(volunteer_one, volunteer_two)][i][0]
                for enu, t in enumerate(self.day_intervals):
                    if t[0] <= s_time <= t[1]:
                        day_number = enu + 1
                        break

                temp_preprocessed_data[cnt] = [volunteer_one, volunteer_two, s_time,
                                               preprocessed_data[(volunteer_one, volunteer_two)][i][1], day_number]
                cnt += 1

        temp_preprocessed_data = temp_preprocessed_data[temp_preprocessed_data[:, 2].argsort()]

        np.savetxt(save_as_text, temp_preprocessed_data, fmt='%d', delimiter=' ')

        return temp_preprocessed_data

    def plot_volunteer_interactions(self, vol_id=None, separate_days=True):
        """
        Plots volunteer interactions as a scatter plot w/ Time on X axis and duration of each interaction on the Y axis.
        :param vol_id: ID of the volunteer. If None, all interactions will be plotted.
        :param separate_days: If True, each day of the data collection will have its own subplot.
        """
        fig_title = f"{self.dataset_name} Data - All Volunteer Interactions"
        if vol_id is not None:
            interactions = self.preprocessed_co_location_data_list[np.where(
                self.preprocessed_co_location_data_list[:, 0:2] == vol_id)[0]]
            fig_title = f"{self.dataset_name} Data - Volunteer ID: {vol_id}"
        else:
            interactions = self.preprocessed_co_location_data_list

        in_contact_interactions = interactions[np.where(interactions[:, 5] == 1)[0], :]
        not_in_contact_interactions = interactions[np.where(interactions[:, 5] == 0)[0], :]

        if separate_days and 6 > len(self.day_intervals) > 1:
            fig, axs = plt.subplots(len(self.day_intervals), 1)
            for i, ax in enumerate(axs):
                d_ind = np.where(not_in_contact_interactions[:, 4] == i + 1)
                p1 = ax.scatter(not_in_contact_interactions[d_ind, 2], not_in_contact_interactions[d_ind, 3], c='red',
                                alpha=0.6)

                d_ind = np.where(in_contact_interactions[:, 4] == i + 1)
                p2 = ax.scatter(in_contact_interactions[d_ind, 2], in_contact_interactions[d_ind, 3], c='blue',
                                alpha=0.6)

                ax.set_title("Day {}".format(i + 1))

            fig.legend((p1, p2), ('Co-Location Only', 'Co-Location in Contact'), 'upper right')
            fig.text(0.5, 0.03, 'Time (s)', ha='center')
            fig.text(0.03, 0.5, 'Length of Interaction (s)', va='center', rotation='vertical')
            fig.suptitle(fig_title, fontsize=16)
        else:
            plt.scatter(not_in_contact_interactions[:, 2], not_in_contact_interactions[:, 3], c='red', alpha=0.6,
                        label="Co-Location Only")
            plt.scatter(in_contact_interactions[:, 2], in_contact_interactions[:, 3], c='blue', alpha=0.6,
                        label="Co-Location in Contact")
            plt.xlabel("Time (s)")
            plt.ylabel("Length of Interaction (s)")
            plt.title(fig_title)
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_contact_duration_histogram(self, vol_id=None):
        fig_title = f"{self.dataset_name} Data - All Volunteer Interactions"
        if vol_id is not None:
            interactions = self.preprocessed_co_location_data_list[np.where(
                self.preprocessed_co_location_data_list[:, 0:2] == vol_id)[0]]
            fig_title = f"{self.dataset_name} Data - Volunteer ID: {vol_id}"
        else:
            interactions = self.preprocessed_co_location_data_list

        in_contact_interactions = interactions[np.where(interactions[:, 5] == 1)[0], :]
        not_in_contact_interactions = interactions[np.where(interactions[:, 5] == 0)[0], :]

        plt.hist(not_in_contact_interactions[:, 3], color='red', alpha=0.5, density=True, label="Co-Location Only")
        plt.hist(in_contact_interactions[:, 3], color='blue', alpha=0.5, density=True, label="Co-Location in Contact")

        plt.xlabel("Event (Contact) Duration (s)")
        plt.ylabel("Probability")
        plt.title(fig_title)
        plt.legend()
        plt.tight_layout()
        plt.show()


    def __check_for_co_location_interactions_in_contact(self):
        if not self.preprocessed_co_location_data_list:
            self.preprocessed_co_location_data_list = []

        for inter_key in self.preprocessed_co_location_data:
            for inter_info in self.preprocessed_co_location_data[inter_key]:
                is_in_contact = 0
                day_number = 0
                if inter_key in self.preprocessed_contact_data:
                    for con_inter_info in self.preprocessed_contact_data[inter_key]:
                        if min(inter_info[0] + inter_info[1], con_inter_info[0] + con_inter_info[1]) - \
                           max(inter_info[0], con_inter_info[0]) > 0:
                            is_in_contact = 1
                            break

                for enu, t in enumerate(self.day_intervals):
                    if t[0] <= inter_info[0] <= t[1]:
                        day_number = enu + 1
                        break

                self.preprocessed_co_location_data_list.append([inter_key[0], inter_key[1], inter_info[0],
                                                                inter_info[1], day_number, is_in_contact])

        self.preprocessed_co_location_data_list = np.array(self.preprocessed_co_location_data_list, dtype=int)


if __name__ == '__main__':
    datasets = ["SFHH", "high_school", "hospital", "primary_school", "workplace_13", "workplace_15"]

    # sfhh = CoLocationData(dataset_name="workplace_15", load_form_pickle=False)
    # Parallel(n_jobs=6)(delayed(CoLocationData)(dataset_name=d, load_form_pickle=False) for d in datasets)

    for d in datasets:
        dat = CoLocationData(dataset_name=d, load_form_pickle=True)
        dat.plot_contact_duration_histogram()
        # cnt = 0
        # for vid in dat.get_volunteer_ids():
        #     # dat.plot_volunteer_interactions(vid)
        #     dat.plot_contact_duration_histogram(vid)
        #
        #     cnt += 1
        #     if cnt == 5:
        #         break
