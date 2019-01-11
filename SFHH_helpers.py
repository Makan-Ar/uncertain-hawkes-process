import numpy as np

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


class SFHHData:
    contact_data_path = "data/SFHH/tij_SFHH.dat_"
    co_location_data_path = "data/SFHH/tij_pres_SFHH.dat"
    # contacts_daily_time_intervals = [(32520, 77580), (115900, 146820)]
    # co_locaiton_daily_time_intervals = [(31500, 71980), (115220, 138040)]
    # start_time = 31500
    # end_time = 146820

    daily_time_intervals = [(31500, 77580), (115220, 146820)]  # (min, max) of both datasets
    interaction_duration = 20
    contact_data = None  # raw SFHH contact data
    co_location_data = None  # raw SFHH co-location data
    preprocessed_contact_data = {}  # volunteer1_id, volunteer2_id, interaction_start_time, duration, day number
    preprocessed_co_location_data = {}  # volunteer1_id, volunteer2_id, interaction_start_time, duration, day number
    __volunteer_ids = None

    def __init__(self, preprocessed=True):
        self.contact_data = np.genfromtxt(self.contact_data_path, dtype=int)
        self.co_location_data = np.genfromtxt(self.co_location_data_path, dtype=int)
        if preprocessed:
            self.run_preprocessing_on_contact_data()
            self.run_preprocessing_on_co_location_data()
            # self.preprocessed_contact_data = np.genfromtxt(self.preprocessed_network_data_path, dtype=int)

    def run_preprocessing_on_contact_data(self, save_as_text=None):
        if len(self.preprocessed_contact_data) > 0:
            return self.preprocessed_contact_data

        return self.__run_preprocessing(self.contact_data, self.preprocessed_contact_data, save_as_text)

    def run_preprocessing_on_co_location_data(self, save_as_text=None):
        if len(self.preprocessed_co_location_data) > 0:
            return self.preprocessed_co_location_data

        return self.__run_preprocessing(self.co_location_data, self.preprocessed_co_location_data, save_as_text)

    def __run_preprocessing(self, raw_data, preprocessed_data, save_as_text=None):
        total_interactions = 0
        for i in range(np.shape(raw_data)[0]):
            inter_key = (raw_data[i, 1], raw_data[i, 2]) if raw_data[i, 1] < raw_data[i, 2] else \
                (raw_data[i, 2], raw_data[i, 1])

            if inter_key not in preprocessed_data:
                preprocessed_data[inter_key] = []

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
                for enu, t in enumerate(self.daily_time_intervals):
                    if t[0] <= s_time <= t[1]:
                        day_number = enu + 1
                        break

                temp_preprocessed_data[cnt] = [volunteer_one, volunteer_two, s_time,
                                               preprocessed_data[(volunteer_one, volunteer_two)][i][1], day_number]
                cnt += 1

        temp_preprocessed_data = temp_preprocessed_data[temp_preprocessed_data[:, 2].argsort()]

        np.savetxt(save_as_text, temp_preprocessed_data, fmt='%d', delimiter=' ')

        return temp_preprocessed_data


if __name__ == '__main__':
    sfhh = SFHHData()
    print(sfhh.preprocessed_contact_data)
    print(sfhh.preprocessed_co_location_data)
    print(len(sfhh.preprocessed_contact_data))
    print(len(sfhh.preprocessed_co_location_data))

