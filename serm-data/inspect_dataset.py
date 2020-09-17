

FILE = './tweets-cikm.txt'
UID_1_FILE = './user_data/tweets-cikm-uid-1.txt'
UID_2_FILE = './user_data/tweets-cikm-uid-2.txt'
LOG = 'inspect-log.txt'

uid_1 = 9836742
uid_2 = 96794283
test_user = 6277272
pid_1_cnt = []
pid_2_cnt = []

uid_1_cnt, uid_2_cnt = 0, 0

f_uid1 = open(UID_1_FILE, 'w')
f_uid2 = open(UID_2_FILE, 'w')
f_log = open(LOG, 'w')

# Print header
f_log.write("ID\tLine\tuid\tpid\n" )

with open(FILE) as fid:
    # with open(self.TWITTER_PATH, 'r') as fid:
    for i, line in enumerate(fid):
        _, uid, _, _, tim, _, _, tweet, pid = line.strip('\r\n').split('')

        if uid == str(uid_1):
            # print("uid_1" + ' ', i, uid)
            f_uid1.write(line)
            uid_1_cnt += 1
            if pid not in pid_1_cnt:
                pid_1_cnt.append(pid)
            f_log.write(str("uid_1" + '\t' + str(i) + '\t' + str(uid)) + '\t' + str(pid) + '\n')
            
        elif uid == str(uid_2):
            # print("uid_2" + ' ', i, uid)
            f_uid2.write(line)
            uid_2_cnt += 1
            if pid not in pid_2_cnt:
                pid_2_cnt.append(pid)
            f_log.write(str("uid_2" + '\t' + str(i) + '\t' + str(uid)) + '\t' + str(pid) +'\n')
        else:
            pass

print("uid_1 entries = {}, uid_2 entries = {}".format(uid_1_cnt, uid_2_cnt))
print("pid_1 entries = {}, pid_2 entries = {}".format(len(pid_1_cnt), len(pid_2_cnt)))

f_log.write("uid_1 entries = {}, uid_2 entries = {}\n".format(uid_1_cnt, uid_2_cnt))
f_log.write("pid_1 entries = {}, pid_2 entries = {}\n".format(len(pid_1_cnt), len(pid_2_cnt)))

f_uid1.close()
f_uid2.close()
f_log.close()

def check_entries(readable):
    uid_cnt = 0
    line_cnt = 0
    unique_pid = []

    with open(readable) as fid:
        for i, line in enumerate(fid):
            _, uid, _, _, tim, _, _, tweet, pid = line.strip('\r\n').split('')
            line_cnt += 1
            if uid == str(uid_1):
                # print("uid_1" + ' ', i, uid)
                # f_uid1.write(line)
                uid_cnt += 1
                if pid not in unique_pid:
                    unique_pid.append(pid)
                
    return uid_cnt, line_cnt, len(unique_pid)


uid, line, pid = check_entries(UID_1_FILE)
print(uid, line, pid)