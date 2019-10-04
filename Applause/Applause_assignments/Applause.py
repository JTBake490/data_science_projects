import numpy as np
import pandas as pd
from itertools import chain

def get_matches(testers, devices, bugs, tester_device):
    testers['lastLogin'] = pd.to_datetime(testers['lastLogin'])

    devices['description'] = devices.description.str.upper()
    devices['description'] = devices.description.str.replace(' ', '')

    count = 0
    country_list = list()
    num_country = input('Please input an integer for how many countries you would like to search, or ALL:  ').upper()
    
    if num_country != 'ALL':
        num_country = int(num_country)
        while count < num_country:
            choice = input('Please input a two letter Country code: ').upper()
            country_list.append(choice)
            count += 1
    else:
        for country in testers.country.unique():
            country_list.append(country)

    country_list = [country.replace(' ', '').strip() for country in country_list]
    for country in country_list:
        if country not in np.array([testers.country]) and country != 'ALL':   #check if there is a tester from specified country
            print(f'No testers in {country}.')
            return None
    
    
    count = 0
    device_list = list()
    num_device = input('Please input an integer for how many devices you would like to search, or ALL:  ').upper()
    if num_device != 'ALL':
        num_device = int(num_device)
        while count < num_device:
            choice = input('Please input a device:  ').upper()
            device_list.append(choice)
            count += 1
    else:
        for device in devices.description.unique():
            device_list.append(device)

    device_list =[device.replace(' ', '').strip() for device in device_list]
    for device in device_list:
        if device not in np.array([devices.description]) and device != 'ALL': #check if device has ever been tested
            print(f'No test for {device}.')
            return None
    
    merge = pd.merge(testers, tester_device, how='outer')  #combine files for easy access
    merge = pd.merge(merge, bugs, how='outer')
    merge['fullName'] = merge['firstName'] + ' ' + merge['lastName'] #create column for full name to cut down on future code.
    
    #If country is ALL and input for device is ALL
    if num_country == 'ALL':
        if num_device == 'ALL':
            device_ids = merge.deviceId.unique()
            tester_ids  = merge.testerId.unique()
            experiences = list()
            testers = list()
            for tester in tester_ids:
                if tester in tester_device.testerId:
                    devices_tested  = set(tester_device[tester_device.testerId==tester].deviceId)
                    tester = merge[merge.testerId==tester].fullName.unique()[0]
                    testers.append(tester)
                    bugs_caught = list(merge[(merge.deviceId==deviceid) & (merge.fullName==tester)].bugId.unique() for deviceid in device_ids)
                    experiences.append((sum(list(len(bugs) for bugs in bugs_caught))))
                    experience = tuple(zip(testers, experiences))
                    experience = sorted(experience, key=lambda tup:(-tup[1], tup[0])) #sort experience in descending order



        #If country is ALL and input for device is one value
        elif num_country == 'ALL' and num_device != 'ALL':
            device_ids = list(chain.from_iterable(list(devices[devices.description == device].deviceId) for device in device_list))
            testers = [list(merge[merge.deviceId==device_id].testerId.unique()) for device_id in device_ids]
            testers = list(set(chain.from_iterable(testers)))
            experiences = list()
            tester_names = list()
            for tester in testers:
                tester = merge[merge.testerId==tester].fullName.unique()[0]
                tester_names.append(tester)
                bugs_caught = list(merge[(merge.deviceId==deviceid) & (merge.fullName==tester)].bugId.unique() for deviceid in device_ids)
                experiences.append((sum(list(len(bugs) for bugs in bugs_caught))))
                experience = tuple(zip(tester_names, experiences))
                experience = sorted(experience, key=lambda tup:(-tup[1], tup[0])) #sort experience in descending order

        for tester in experience[::-1]:
            if tester[1]==0:
                experience.remove(tester)

        for idx, device_testers in enumerate(experience):
            yield f'{device_testers[0]} => {device_testers[1]}'
    


#     If input for country is not ALL and device is not ALL
    if num_country != 'ALL':
        if num_device != 'ALL':
            device_ids = list(chain.from_iterable(list(devices[devices.description == device].deviceId) for device in device_list))
            testers = [list(merge[merge.country==country].testerId.unique()) for country in country_list]
            testers = list(set(chain.from_iterable(testers)))
            experiences = list()
            tester_names = list()

            for tester in testers:
                tester = merge[merge.testerId==tester].fullName.unique()[0]
                tester_names.append(tester)
                bugs_caught = list(merge[(merge.deviceId==deviceid) & (merge.fullName==tester)].bugId.unique() for deviceid in device_ids)
                experiences.append((sum(list(len(bugs) for bugs in bugs_caught))))
                experience = tuple(zip(tester_names, experiences))
                experience = sorted(experience, key=lambda tup:(-tup[1], tup[0])) # sort experience in descending order
        else:
    #     If input for country is not ALL and device is ALL
            tester_ids = list(chain.from_iterable(list(merge[merge.country==country].testerId.unique()) for country in country_list)) #retrieve all tester IDs for specified country
            experiences = list() 
            tester_names = list()     

            for tester in tester_ids:
                if tester in np.array([tester_device.testerId]):
                    device_ids = set(tester_device[tester_device.testerId==tester].deviceId)
                    tester_name = merge[merge.testerId==tester].fullName.unique()[0]
                    tester_names.append(tester_name)
                    
                    bugs_caught = np.array([list(merge[(merge.deviceId==deviceid) & (merge.fullName==tester_name)].bugId.unique()) for deviceid in device_ids])
                    experiences.append((sum(list(len(bugs) for bugs in bugs_caught))))
            experience = tuple(zip(tester_names, experiences))
            experience = sorted(experience, key=lambda tup:(-tup[1], tup[0])) # sort in expereience in descending order

        for tester in experience[::-1]:
            if tester[1]==0:
                experience.remove(tester)

        for idx, device_testers in enumerate(experience):
            yield f'{device_testers[0]} => {device_testers[1]}'


if __name__ == '__main__':
    bugs = pd.read_csv('../Applause_files/bugs.csv')                   # all bugs that were reported; gives the device and tester
    devices = pd.read_csv('../Applause_files/devices.csv')             # the names of each device
    testers = pd.read_csv('../Applause_files/testers.csv')             # tester information; does not include devices
    tester_device = pd.read_csv('../Applause_files/tester_device.csv') # which devices each tester tested



    
    print(list(get_matches(testers, devices, bugs, tester_device)))