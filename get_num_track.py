import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='get_num_track')
    parser.add_argument("--res_seq_path",  type=str, default='output/mot17/val/MOT17-13-SDP.txt')
    args = parser.parse_args()
    return args

args=parse_args()
res_seq=open(args.res_seq_path)
l=[]
lines=res_seq.readlines()
for line in lines:
    s=line.split(',')
    l.append(s[1])
print(len(set(l)))





