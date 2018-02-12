import numpy as np

def LearnSuc():
    
    typenames = ['author','conf','keyword','ref']

    n_iter = 50
    n_neg = 10
    n_vec = 3
    alpha = 0.01 # learning rate
    INITFLOAT = 0.1
    w_type = [1,1,1,1]
    n_type = len(w_type)
    list_item2vec = [{} for i in range(n_type)]

    authorid2authorname = {}
    fr = open('small/ItemAuthor.txt','rb')
    for line in fr:
        arr = line.strip('\r\n').split('\t')
        authorid,authorname = arr[0],arr[1]
        authorid2authorname[authorid] = authorname
    fr.close()
    confid2confname = {}
    fr = open('small/ItemConference.txt','rb')
    for line in fr:
        arr = line.strip('\r\n').split('\t')
        confid,confname = arr[0],arr[1]
        confid2confname[confid] = confname
    fr.close()
    paperid2titleyear = {}
    fr = open('small/ItemPaper.txt','rb')
    for line in fr:
        arr = line.strip('\r\n').split('\t')
        paperid,titleyear = arr[0],arr[1]+','+arr[2]
        paperid2titleyear[paperid] = titleyear
    fr.close()

    # load data
    behaviors = []
    fr = open('small/PaperItemsets.txt','rb')
    for line in fr:
        arr = line.strip('\r\n').split('\t')        
        behavior = []
        paperid = arr[0]        
        for i in range(n_type):
            if arr[i+1] == '':
                behavior.append([])
            else:
                behavior.append(arr[i+1].split(','))
        behaviors.append([paperid,behavior])
    fr.close()
    n_behavior = len(behaviors)

    # initialization
    for [paperid,behavior] in behaviors:
        for i in range(n_type):
            for item in behavior[i]:
                if not item in list_item2vec[i]:
                    list_item2vec[i][item] = (2.0*np.random.rand(n_vec)-1.0)*INITFLOAT
    list_nitem = [len(list_item2vec[i]) for i in range(n_type)]
    behaviors_neg = []
    for b in range(n_behavior):
        behavior = behaviors[b][1]
        behavior_neg = []
        for i in range(n_type):
            itemids = np.arange(list_nitem[i])
            np.random.shuffle(itemids)
            items = list_item2vec[i].keys()
            items = [items[itemids[j]] for j in range(len(behavior[i]))]
            behavior_neg.append(items)
        behaviors_neg.append(behavior_neg)

    fw_evaluate = open('evaluate.txt','w')
    s = 'iter\tmean_r_pos\tmean_r_neg\tmean_r_diff\tmean_norm_pos\tmean_norm_neg\tmean_norm_diff' \
            +'\tmax_r_pos\tmax_r_neg\tmax_norm_pos\tmax_norm_neg\tmin_r_pos\tmin_r_neg\tmin_norm_pos\tmin_norm_neg'
    fw_evaluate.write(s+'\n')
    print s

    # main 
    for _iter in range(n_iter):

        # evaluate
        norm_b_pos,norm_b_neg = [],[]
        r_b_pos,r_b_neg = [],[]
        for [paperid,behavior] in behaviors:
            vec_b = np.zeros(n_vec)
            for i in range(n_type):
                for item in behavior[i]:
                    vec_b += w_type[i]*list_item2vec[i][item]
            norm_b = np.linalg.norm(vec_b,2)
            norm_b_pos.append(norm_b)
            r_b = EasyTanh(norm_b/2)
            r_b_pos.append(r_b)
        for behavior in behaviors_neg:
            vec_b = np.zeros(n_vec)
            for i in range(n_type):
                for item in behavior[i]:
                    vec_b += w_type[i]*list_item2vec[i][item]
            norm_b = np.linalg.norm(vec_b,2)
            norm_b_neg.append(norm_b)            
            r_b = EasyTanh(norm_b/2)
            r_b_neg.append(r_b)
        mean_norm_b_pos = np.mean(norm_b_pos)
        mean_norm_b_neg = np.mean(norm_b_neg)
        mean_r_b_pos = np.mean(r_b_pos)
        mean_r_b_neg = np.mean(r_b_neg)
        max_norm_b_pos = np.max(norm_b_pos)
        max_norm_b_neg = np.max(norm_b_neg)
        max_r_b_pos = np.max(r_b_pos)
        max_r_b_neg = np.max(r_b_neg)
        min_norm_b_pos = np.min(norm_b_pos)
        min_norm_b_neg = np.min(norm_b_neg)
        min_r_b_pos = np.min(r_b_pos)
        min_r_b_neg = np.min(r_b_neg)
        s = str(_iter)+'\t'+str(mean_r_b_pos)+'\t'+str(mean_r_b_neg)+'\t'+str(mean_r_b_pos-mean_r_b_neg) \
                +'\t'+str(mean_norm_b_pos)+'\t'+str(mean_norm_b_neg)+'\t'+str(mean_norm_b_pos-mean_norm_b_neg) \
                +'\t'+str(max_r_b_pos)+'\t'+str(max_r_b_neg)+'\t'+str(max_norm_b_pos)+'\t'+str(max_norm_b_neg) \
                +'\t'+str(min_r_b_pos)+'\t'+str(min_r_b_neg)+'\t'+str(min_norm_b_pos)+'\t'+str(min_norm_b_neg)
        fw_evaluate.write(s+'\n')
        print s

        # output item2vec
        fw_item2vec = open('item2vec/item2vec_'+str(_iter)+'.txt','w')
        for i in range(n_type):
            for item in list_item2vec[i]:
                s = ''
                for j in range(n_vec):
                    s += '\t'+str(list_item2vec[i][item][j])
                name = item
                if i == 0 and item in authorid2authorname: name = authorid2authorname[item]
                if i == 1 and item in confid2confname: name = confid2confname[item]
                fw_item2vec.write(str(i)+'\t'+typenames[i]+'\t'+str(item)+'\t'+name+s+'\n')
        fw_item2vec.close()

        # output behavior2vec
        label_r_norm_paperid_bstr_bvec = []
        for [paperid,behavior] in behaviors:
            vec_b = np.zeros(n_vec)
            for i in range(n_type):
                for item in behavior[i]:
                    vec_b += w_type[i]*list_item2vec[i][item]
            norm_b = np.linalg.norm(vec_b,2)
            r_b = EasyTanh(norm_b/2)
            bstr = GetBehaviorStr(behavior,paperid,authorid2authorname,confid2confname,paperid2titleyear)
            label_r_norm_paperid_bstr_bvec.append(['+',r_b,norm_b,paperid,bstr,vec_b])
        for behavior in behaviors_neg:
            vec_b = np.zeros(n_vec)
            for i in range(n_type):
                for item in behavior[i]:
                    vec_b += w_type[i]*list_item2vec[i][item]
            norm_b = np.linalg.norm(vec_b,2)
            r_b = EasyTanh(norm_b/2)
            bstr = GetBehaviorStr(behavior,'',authorid2authorname,confid2confname,paperid2titleyear)
            label_r_norm_paperid_bstr_bvec.append(['-',r_b,norm_b,'null',bstr,vec_b])
        fw_behavior2vec = open('behavior2vec/behavior2vec_'+str(_iter)+'.txt','w')
        for [label,r,norm,paperid,bstr,bvec] in sorted(label_r_norm_paperid_bstr_bvec,key=lambda x:-x[2]):
            s = ''
            for j in range(n_vec):
                s += '\t'+str(bvec[j])
            s = str(label)+'\t'+str(r)+'\t'+str(norm)+'\t'+paperid+'\t'+str(bstr)+s
            fw_behavior2vec.write(s+'\n')
        fw_behavior2vec.close()

        # optimization
        bids = np.arange(n_behavior)
        np.random.shuffle(bids)
        for b in range(n_behavior):
            # positive
            behavior = behaviors[b][1]
            vec_b = np.zeros(n_vec)
            for i in range(n_type):
                for item in behavior[i]:
                    vec_b += w_type[i]*list_item2vec[i][item]
            norm_b = np.linalg.norm(vec_b,2)
            vec_inc = EasyPosSinh(norm_b)*vec_b
            for i in range(n_type):
                for item in behavior[i]:
                    list_item2vec[i][item] += alpha*w_type[i]*vec_inc
            # negative
            for _neg in range(n_neg):
                behavior_neg = []
                for i in range(n_type):
                    itemids = np.arange(list_nitem[i])
                    np.random.shuffle(itemids)
                    items = list_item2vec[i].keys()
                    items = [items[itemids[j]] for j in range(len(behavior[i]))]
                    behavior_neg.append(items)
                vec_b_neg = np.zeros(n_vec)
                for i in range(n_type):
                    for item in behavior_neg[i]:
                        vec_b_neg += w_type[i]*list_item2vec[i][item]
                norm_b_neg = np.linalg.norm(vec_b_neg,2)
                vec_inc_neg = EasyNegSinh(norm_b_neg)*vec_b_neg
                for i in range(n_type):
                    for item in behavior_neg[i]:
                        list_item2vec[i][item] -= alpha*w_type[i]*vec_inc_neg

    fw_evaluate.close()

if __name__ == '__main__':
    LearnSuc()

