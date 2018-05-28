/*
 *  This is the performance implementation of LearnSUC model.
 *
 *  @author Daheng Wang
 *  @email  dwang8@nd.edu
 *
 *  Publication: Multi-Type Itemset Embedding for Learning Behavior Success
 *  Authors: Daheng Wang, Meng Jiang, Qingkai Zeng, Zachary Eberhart, Nitesh Chawla
 *  Organization: University of Notre Dame, Notre Dame, Indiana, 46556, USA
 *
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <random>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <array>
#include <functional>
//#include <cmath>
//#include <thread>

#define MAX_STR_LEN 100

using namespace std;

/* Precisions */
typedef long long int lint;
typedef double real;

/* Item, type, and behavior information */
lint items_num, types_num, behaviors_num;
vector<lint> items, types, behaviors;
map<lint, lint> item2type, item2item_idx, type2type_idx, item_idx2type_idx, behavior2behavior_idx;
vector<vector<lint>> type_idx2item_indices, behavior_idx2item_indices;
vector<real> type_weights, behavior_rates;

/* Embeddings */
vector<vector<real>> item_embeddings;

/* Parameters */
char itemlist_file[MAX_STR_LEN], behaviorlist_file[MAX_STR_LEN], output_file[MAX_STR_LEN],
        typeweights_file[MAX_STR_LEN] = "", behaviorrates_file[MAX_STR_LEN] = "";
lint dimensions_num  = 128, threads_num = 8, total_samples_num = (lint)1e6,
        negative_sampling_num = 10, negative_sampling_mode = 1;
real learning_rate = 0.025;

/* Misc */
lint total_samples_num_unit = (lint)1e6;
real total_samples_num_input = 1;
pthread_mutex_t cout_mutex = PTHREAD_MUTEX_INITIALIZER;
lint curr_samples_num = 0;
real curr_learning_rate = learning_rate;


/*
 * Compute behavior vector Euclidean L2 norm
 */
real get_b_norm(vector<real>& b_vec) {
    real b_norm = 0;
    for (const auto& d: b_vec) {
        b_norm += pow(d, 2);
    }
    return sqrt(b_norm);
}


/*
 * Compute coefficient for positive behavior in equation (9)
 */
real quick_pos_b_sinh(real& pos_b_norm, int optimization=1) {
    switch(optimization) {
        case 1:
            if (pos_b_norm > 10) {
                return 0;
            } else if (pos_b_norm < 0.1) {
                return 100;
            } else {
                return 1/pos_b_norm/sinh(pos_b_norm);
            }
        default:
            return 1/pos_b_norm/sinh(pos_b_norm);
    }
}


/*
 * Compute coefficient for negative behavior in equation (9)
 */
real quick_neg_b_sinh(real& neg_b_norm, int optimization=1) {
    switch(optimization) {
        case 1:
            if (neg_b_norm < 0.05 or neg_b_norm > 100) {
                return 0;
            } else {
                return 1/neg_b_norm/neg_b_norm/neg_b_norm/sinh(1/neg_b_norm);
            }
        default:
            return 1/neg_b_norm/neg_b_norm/neg_b_norm/sinh(1/neg_b_norm);
    }
}


/*
 * Thread for training LearnSUC model
 */
void *train_learn_suc_thread(void *) {
    //auto th_id = (long)thread_id;
    //this_thread::sleep_for(std::chrono::milliseconds(100*th_id));
    auto checkpoints_interval = (lint)1000;
    lint thread_samples = 0, checkpoint_samples = 0;

    //random_device rd;
    //mt19937 engine(rd());

    //DEBUG
    random_device rd;
    array<int, mt19937::state_size> seed_data;
    generate_n(seed_data.data(), seed_data.size(), ref(rd));
    seed_seq seq(begin(seed_data), end(seed_data));
    mt19937 engine(seq);

    //auto job_samples = (real)total_samples_num/threads_num/checkpoints_interval;
    //job_samples = ceil(job_samples) * checkpoints_interval;
    auto job_samples = total_samples_num / threads_num + 1;
    for (thread_samples = 0; thread_samples <= job_samples; thread_samples ++) {
        /*
         * Sample positive behavior
         * */
        /* Sample a positive behavior index */
        uniform_int_distribution<lint> dist_b_size(0, behaviors_num-1);
        auto sampled_pos_b_idx = dist_b_size(engine);
        vector<lint> pos_b_t_i_counts(static_cast<unsigned long>(types_num));
        auto pos_b_size = behavior_idx2item_indices[sampled_pos_b_idx].size();
        const auto& pos_b_observed_rate = behavior_rates[sampled_pos_b_idx];

        /* Compute positive behavior vector (type-weighted sum of item vectors) */
        vector<real> pos_b_vec(static_cast<unsigned long>(dimensions_num));
        for(const auto& i_idx: behavior_idx2item_indices[sampled_pos_b_idx]) {
            const auto& t_idx = item_idx2type_idx[i_idx];
            const auto& t_w = type_weights[t_idx];
            pos_b_t_i_counts[t_idx] ++;

            for(lint d=0; d<dimensions_num; ++d) {
                pos_b_vec[d] += item_embeddings[i_idx][d] * t_w;
            }
        }

        /* Compute positive behavior norm, gradient coefficient */
        auto pos_b_norm = get_b_norm(pos_b_vec);
        auto pos_b_sinh = quick_pos_b_sinh(pos_b_norm);

        /* Compute gradient (with out w_t_(c)) */
        for(lint d=0; d<dimensions_num; ++d) {
            pos_b_vec[d] *= (pos_b_sinh * pos_b_observed_rate);
        }

        /* Update item embeddings */
        for(const auto& i_idx: behavior_idx2item_indices[sampled_pos_b_idx]) {
            const auto& t_w = type_weights[item_idx2type_idx[i_idx]];
            for(lint d=0; d<dimensions_num; ++d) {
                item_embeddings[i_idx][d] += pos_b_vec[d] * t_w * curr_learning_rate;
            }
        }

        /*
         * Sample negative behaviors
         * If only positive behaviors are observed (no behaviorrates_file given)
         * */
        if (strlen(behaviorrates_file) == 0) {
            for (lint n = 0; n < negative_sampling_num; ++n) {
                /* Sample a negative behavior based on given negative sampling strategy */
                set<lint> neg_b_i_indices;
                unsigned long cum_neg_b_i_count = 0;

                if (negative_sampling_mode == 1) {
                    /* Size-constrained negative behavior sampling */
                    // Generate samples number of each item type
                    // Randomly cut positive behavior size into types number parts
                    vector<lint> neg_b_t_cuts(static_cast<unsigned long>(types_num - 1));
                    uniform_int_distribution<lint> dist_pos_b_size(0, (lint)(pos_b_size - 1));
                    for (lint c = 0; c < types_num - 1; ++c) {
                        neg_b_t_cuts[c] = dist_pos_b_size(engine);
                    }

                    sort(neg_b_t_cuts.begin(), neg_b_t_cuts.end());  // Sort cuts ascending
                    // Count items number between cuts
                    vector<lint> neg_b_t_i_counts(static_cast<unsigned long>(types_num));
                    for (lint t = 0; t < types_num - 1; ++t) {
                        if (t == 0) {
                            neg_b_t_i_counts[0] = neg_b_t_cuts[0];
                        } else {
                            neg_b_t_i_counts[t] = neg_b_t_cuts[t] - neg_b_t_cuts[t - 1];
                        }
                    }
                    neg_b_t_i_counts.back() = (lint)pos_b_size - neg_b_t_cuts.back();

                    // Generate random item indices in each type
                    for (const auto &t: types) {
                        const auto &type_idx = type2type_idx[t];
                        const auto &neg_b_t_i_count = neg_b_t_i_counts[type_idx];
                        cum_neg_b_i_count += neg_b_t_i_count;
                        // Sample type items without duplicates
                        uniform_int_distribution<lint> dist_t_i_size(0, (lint)(type_idx2item_indices[type_idx].size() - 1));
                        while (neg_b_i_indices.size() < cum_neg_b_i_count) {
                            const auto &sampled_t_i_idx = type_idx2item_indices[type_idx][dist_t_i_size(engine)];
                            neg_b_i_indices.insert(sampled_t_i_idx);
                        }
                    }
                } else if (negative_sampling_mode == 2) {
                    /* Type-distribution constrained negative behavior sampling */
                    for (const auto &t: types) {
                        const auto &type_idx = type2type_idx[t];
                        const auto &pos_b_t_i_count = pos_b_t_i_counts[type_idx];
                        cum_neg_b_i_count += pos_b_t_i_count;
                        // Sample type items without duplicates
                        uniform_int_distribution<lint> dist_t_i_size(0, (lint)(type_idx2item_indices[type_idx].size() - 1));
                        while (neg_b_i_indices.size() < cum_neg_b_i_count) {
                            const auto &sampled_t_i_idx = type_idx2item_indices[type_idx][dist_t_i_size(engine)];
                            neg_b_i_indices.insert(sampled_t_i_idx);
                        }
                    }
                } else {
                    cout << "Error: unknown negative sampling mode!" << endl;
                    exit(1);
                }

                /* Compute negative behavior vector (type-weighted sum of item vectors) */
                vector<real> neg_b_vec(static_cast<unsigned long>(dimensions_num));
                for (const auto &i_idx: neg_b_i_indices) {
                    const auto &t_w = type_weights[item_idx2type_idx[i_idx]];
                    for (lint d = 0; d < dimensions_num; ++d) {
                        neg_b_vec[d] += item_embeddings[i_idx][d] * t_w;
                    }
                }

                /* Compute negative behavior norm, gradient coefficient */
                auto neg_b_norm = get_b_norm(neg_b_vec);
                auto neg_b_sinh = quick_neg_b_sinh(neg_b_norm);

                /* Compute gradient vector (with out w_t_(c)) */
                for (lint d = 0; d < dimensions_num; ++d) {
                    neg_b_vec[d] *= neg_b_sinh;
                }

                /* Update item embeddings */
                for (const auto &i_idx: neg_b_i_indices) {
                    const auto &t_w = type_weights[item_idx2type_idx[i_idx]];
                    for (lint d = 0; d < dimensions_num; ++d) {
                        item_embeddings[i_idx][d] -= neg_b_vec[d] * t_w * curr_learning_rate;
                    }
                }
            }
        }

        /* Check for checkpoint */
        if (thread_samples - checkpoint_samples == checkpoints_interval || thread_samples == job_samples) {
            /* Update checkpoint */
            curr_samples_num += checkpoints_interval;
            checkpoint_samples = thread_samples;

            /* Report progress */
            // Use mutex to avoid corrupting cout by multiple threads
            // May slight affect efficiency if checkpoints_interval is too small
            pthread_mutex_lock(&cout_mutex);
            auto prog = (real)curr_samples_num/total_samples_num*100;
            cout << fixed << setw(10) << setprecision(6)
                 << "Current learning rate: " << curr_learning_rate << "; "
                 << fixed << setw(5) << setprecision(2)
                 << "Progress: " << prog << "%\r";
            cout.flush();
            pthread_mutex_unlock(&cout_mutex);

            /* Update learning rate */
            auto next_learning_rate = learning_rate * (1 - ((real)curr_samples_num/total_samples_num));
            if (next_learning_rate < learning_rate * 0.0001) {
                // Set minial learning rate
                curr_learning_rate = learning_rate * 0.0001;
            } else {
                curr_learning_rate = next_learning_rate;
            }
        }
    }
    pthread_exit(nullptr);
}


/*
 * Train LearnSUC model by multi-threading
 */
void train_learn_suc() {
    long thread_id;
    pthread_t threads[threads_num];

    cout << "Start training LearnSUC model..." << endl;
    for (thread_id=0; thread_id<threads_num; ++thread_id) {
        //pthread_create(&threads[thread_id], nullptr, train_learn_suc_thread, (void *)thread_id);
        pthread_create(&threads[thread_id], nullptr, train_learn_suc_thread, nullptr);
    }
    for (thread_id=0; thread_id<threads_num; ++thread_id) {
        pthread_join(threads[thread_id], nullptr);
    }
    cout << endl << "Done!" << endl;
}

/*
 * Read in items information from itemlist file.
 * Each line follows format: <non_neg_int_item>\t<non_neg_int_item_type>
 */
void read_itemlist_file(const string& itemlist_file, char delim='\t') {
    cout << "Reading itemlist file..." << endl;
    ifstream filein(itemlist_file);
    if (!filein) {
        cout << "Error: file of itemlist not found!" << endl;
        exit(1);
    }

    set<lint> types_set;
    /* Populate items, item2type, item2item_idx */
    lint item_idx_count = 0;
    for (string line; getline(filein, line); ) {  // item_idx follow input items order
        vector<lint> tokens;
        stringstream ss(line);
        string token;
        while (getline(ss, token, delim)) {
            tokens.push_back(stol(token));  // Parse to lint
        }
        if(tokens.size() != 2) {  // Line format wrong
            cout << "Error: input itemlist file format wrong!" << endl;
            exit(1);
        } else {
            items.push_back(tokens[0]);
            item2type.insert(make_pair(tokens[0], tokens[1]));
            item2item_idx.insert(make_pair(tokens[0], item_idx_count));
            item_idx_count ++;
            types_set.insert(tokens[1]);
        }
    }
    items_num = (lint)item2type.size();

    cout << "Indexing item types..."<< endl;
    /* Populate types, type2type_idx */
    lint type_idx_count = 0;
    for (const auto& t: types_set) {
        types.push_back(t);
        type2type_idx.insert(make_pair(t, type_idx_count));
        type_idx_count ++;
    }
    types_num = (lint)type2type_idx.size();

    multimap<lint, lint> type_idx22item_idx;
    /* Populate item_idx2type_idx */
    for (const auto& i_t_pair : item2type) {
        item_idx2type_idx.insert(make_pair(item2item_idx[i_t_pair.first], type2type_idx[i_t_pair.second]));
        type_idx22item_idx.insert(make_pair(type2type_idx[i_t_pair.second], item2item_idx[i_t_pair.first]));
    }

    /* Populate type_idx2item_indices */
    for (const auto& t: types_set) {
        vector<lint> item_indices;
        auto range = type_idx22item_idx.equal_range(type2type_idx[t]);
        for (auto& r=range.first; r!=range.second; ++r) {
            item_indices.push_back(r->second);
        }
        type_idx2item_indices.push_back(item_indices);
    }

    cout << "Done!\t" << "#items: " << items_num << "; #item types: " << types_num
         << "; Distribution (type-#): ";
    for (const auto& t: types) {
        const auto& type_idx = type2type_idx[t];
        cout << t << "-" << type_idx2item_indices[type_idx].size() << " ";
    }
    cout << endl;
}


/*
 * Read in behaviors information from behaviorlist file.
 * Each line follows format: <non_neg_int_behavior>\t<non_neg_int_item1>[,<non_neg_int_item2>,...]
 */
void read_behaviorlist_file(const string& behaviorlist_file, char delim_l1='\t', char delim_l2=',') {
    cout << "Reading behaviorlist file..." << endl;
    ifstream filein(behaviorlist_file);
    if (!filein) {
        cout << "Error: file of behaviorlist not found!" << endl;
        exit(1);
    }

    /* Populate behaviors, behavior2behavior_idx, behavior_idx2item_indices */
    set<lint> items_set(items.begin(), items.end());
    bool in_items_set;
    lint behavior_idx_count = 0;
    for (string line; getline(filein, line); ) {
        vector<string> tokens;
        stringstream ss(line);
        string token;
        while (getline(ss, token, delim_l1)) {
            tokens.push_back(token);
        }
        if(tokens.size() != 2) {
            cout << "Error: input behaviorlist file format wrong!" << endl;
            exit(1);
        } else {
            behaviors.push_back(stol(tokens[0]));
            behavior2behavior_idx.insert(make_pair(stol(tokens[0]), behavior_idx_count));

            vector<lint> tokens_l2;
            stringstream ss_l2(tokens[1]);  // tokens[0] for behavior; tokens[1] for items
            string token_l2;
            while (getline(ss_l2, token_l2, delim_l2)) {
                /* Discard items not in itemlist file */
                in_items_set = items_set.find(stol(token_l2)) != items_set.end();
                if(in_items_set) {
                    tokens_l2.push_back(item2item_idx[stol(token_l2)]);
                }
            }
            behavior_idx2item_indices.push_back(tokens_l2);

            behavior_idx_count ++;
        }
    }
    behaviors_num = (lint)behaviors.size();
    cout << "Done!\t" << "#behaviors: " << behaviors_num << endl;
}

/*
 * Read in item type weights information from typeweigths file.
 * Each line follows format: <non_neg_int_item_type>\t<pos_float_item_type_weight>
 */
void read_typeweights_file (const string& typeweights_file, char delim='\t') {
    cout << "Reading typeweights file..." << endl;
    ifstream filein(typeweights_file);
    if (!filein) {
        cout << "Error: file of typeweights not found!" << endl;
        exit(1);
    }

    set<lint> types_set(types.begin(), types.end());
    bool in_types_set;
    /* Update type_weights */
    for (string line; getline(filein, line); ) {
        vector<real> tokens;
        stringstream ss(line);
        string token;
        while (getline(ss, token, delim)) {
            tokens.push_back(stod(token));  // Parse to real
        }
        if(tokens.size() != 2) {  // Line format wrong
            cout << "Error: input typeweights file format wrong!" << endl;
            exit(1);
        } else {
            /* Discard types not in itemlist file */
            in_types_set = types_set.find((lint)tokens[0]) != types_set.end();
            if (in_types_set) {
                const auto& type_idx = type2type_idx[(lint)tokens[0]];
                type_weights[type_idx] = tokens[1];
            }
        }
    }
    cout << "Done!" << endl;
}

/*
 * Read in positive behavior success rates information from behaviorrates file.
 * Each line follows format: <non_neg_int_behavior>\t<pos_float_behavior_success_rate>
 */
void read_behaviorrates_file (const string& behaviorrates_file, char delim='\t') {
    cout << "Reading behaviorrates file..." << endl;
    ifstream filein(behaviorrates_file);
    if (!filein) {
        cout << "Error: file of behaviorrates not found!" << endl;
        exit(1);
    }

    set<lint> behaviors_set(behaviors.begin(), behaviors.end());
    bool in_behaviors_set;
    /* Update behavior_rates */
    for (string line; getline(filein, line); ) {
        vector<real> tokens;
        stringstream ss(line);
        string token;
        while (getline(ss, token, delim)) {
            tokens.push_back(stod(token));  // Parse to real
        }
        if(tokens.size() != 2) {  // Line format wrong
            cout << "Error: input typeweights file format wrong!" << endl;
            exit(1);
        } else {
            /* Discard behaviors not in behaviorlist file */
            in_behaviors_set = behaviors_set.find((lint)tokens[0]) != behaviors_set.end();
            if (in_behaviors_set) {
                const auto& behavior_idx = behavior2behavior_idx[(lint)tokens[0]];
                behavior_rates[behavior_idx] = tokens[1];
            }
        }
    }
    cout << "Done!" << endl;
}


/*
 * Initialization operations
 */
void initialize() {
    /* total_samples_num */
    total_samples_num = (lint)(total_samples_num_unit * total_samples_num_input);

    /* Item type weights */
    // Default weight scheme: all equal to 1
    real default_weight  = 1.0;
    type_weights = vector<real>(static_cast<unsigned long>(types_num), default_weight);
    // If optional typeweights file provided
    if (strlen(typeweights_file) != 0) {
        read_typeweights_file(typeweights_file);
    }
    cout << "Weight scheme (type-weight): ";
    for (const auto& t: types) {
        const auto& type_idx = type2type_idx[t];
        cout << t << "-" << type_weights[type_idx] << " ";
    }
    cout << endl;

    /* Positive behavior success rates */
    // Default behavior success rate is 1
    real default_rate  = 1.0;
    behavior_rates = vector<real>(static_cast<unsigned long>(behaviors_num), default_rate);
    // If optional behaviorrates file provided
    if (strlen(behaviorrates_file) != 0) {
        read_behaviorrates_file(behaviorrates_file);
    }

    cout << "Initializing..." << endl;
    /* Item embeddings */
    random_device rd;
    // Random engines
    mt19937 engine(rd());
    // knuth_b engine(rd());
    // default_random_engine engine(rd()) ;

    real default_low_bound = -1e-3;
    real default_high_bound = 1e-3;
    uniform_real_distribution<real> dist(default_low_bound, default_high_bound);

    vector<real> item_emb(static_cast<unsigned long>(dimensions_num));
    for (lint i=0; i<items_num; ++i) {
        for (lint d=0; d<dimensions_num; d++) {
            item_emb[d] = dist(engine);
        }
        item_embeddings.push_back(item_emb);
    }
    cout << "Done!" << endl;
}


/*
 * Write out item embeddings to target file
 */
void output_embeddings(const string& output_file){
    cout << "Writing out item embeddings..." << endl;
    ofstream fileout(output_file);
    /* Write header line */
    fileout << items_num << "\t" << dimensions_num << endl;
    /* Write embeddings */
    for (lint i=0; i<items_num; ++i) {
        fileout << items[i] << "\t";
        for (lint d=0; d<dimensions_num-1; ++d) {
            fileout << item_embeddings[i][d] << "\t";
        }
        fileout << item_embeddings[i].back() << endl;
    }
    cout << "Done!" << endl;
}

/*
 * Parse command line arguments
 */
int parse_args(char *str, int argc, char **argv) {
    int a;
    for (a=1; a<argc; a++) {
        if (!strcmp(str, argv[a])) {
            if (a == argc-1) {
                cout << "Argument missing for " << str << endl;
                exit(1);
            }
            return a;
        }
    }
    return -1;
}


int main(int argc, char **argv) {
    cout << "====================================================" << endl;
    /* Parse arguments */
    int i;
    // Required
    if ((i = parse_args((char *)"--itemlist", argc, argv)) > 0) strcpy(itemlist_file, argv[i + 1]);
    if ((i = parse_args((char *)"--behaviorlist", argc, argv)) > 0) strcpy(behaviorlist_file, argv[i + 1]);
    if ((i = parse_args((char *)"--output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    // Required with defaults
    if ((i = parse_args((char *)"--dim", argc, argv)) > 0) dimensions_num = stol(argv[i + 1]);
    if ((i = parse_args((char *)"--mode", argc, argv)) > 0) negative_sampling_mode = stol(argv[i + 1]);
    if ((i = parse_args((char *)"--samples", argc, argv)) > 0) total_samples_num_input = stod(argv[i + 1]);
    if ((i = parse_args((char *)"--negative", argc, argv)) > 0) negative_sampling_num = stol(argv[i + 1]);
    if ((i = parse_args((char *)"--rate", argc, argv)) > 0) learning_rate = stod(argv[i + 1]);
    if ((i = parse_args((char *)"--threads", argc, argv)) > 0) threads_num = stol(argv[i + 1]);
    // Optional
    if ((i = parse_args((char *)"--typeweights", argc, argv)) > 0) strcpy(typeweights_file, argv[i + 1]);
    if ((i = parse_args((char *)"--behaviorrates", argc, argv)) > 0) strcpy(behaviorrates_file, argv[i + 1]);

    cout << "Arguments:" << endl
         << " --itemlist: " << itemlist_file << endl
         << " --behaviorlist: " << behaviorlist_file << endl
         << " --output: " << output_file << endl
         << " --dim: " << dimensions_num << endl
         << " --mode: " << negative_sampling_mode << endl
         << " --samples (Million): " << total_samples_num_input << endl
         << " --negative: " << negative_sampling_num << endl
         << " --rate: " << learning_rate << endl
         << " --threads: " << threads_num << endl;
    if (strlen(typeweights_file) != 0) {
        cout << " --typeweights: " << typeweights_file << endl;
    }
    if (strlen(behaviorrates_file) != 0) {
        cout << " --behaviorrates: " << behaviorrates_file << endl;
    }

    cout << "====================================================" << endl;
    using clock = chrono::steady_clock;
    /* Read in itemlist, behaviorlist files and initialization */
    auto t_i_s = clock::now();
    read_itemlist_file(itemlist_file);
    read_behaviorlist_file(behaviorlist_file);
    initialize();
    auto t_i_e = clock::now();
    cout << "====================================================" << endl;

    /* Train LearnSUC use multi-threading */
    auto t_t_s = clock::now();
    train_learn_suc();
    auto t_t_e = clock::now();
    cout << "====================================================" << endl;

    /* Output embedding */
    auto t_o_s = clock::now();
    output_embeddings(output_file);
    auto t_o_e = clock::now();
    cout << "====================================================" << endl;

    /* Print summary */
    auto init_time = (real)chrono::duration_cast<chrono::milliseconds>(t_i_e-t_i_s).count();
    auto training_time = (real)chrono::duration_cast<chrono::milliseconds>(t_t_e-t_t_s).count();
    auto output_time = (real)chrono::duration_cast<chrono::milliseconds>(t_o_e-t_o_s).count();
    auto total_time = (real)chrono::duration_cast<chrono::milliseconds>(t_o_e-t_i_s).count();
    cout << fixed << setw(5) << setprecision(2) << "Total elapsed time: " << total_time/1000 << " s" << endl
         << " - Initialization: " << init_time/1000 << " s"
            << " (" << (init_time/total_time)*100 << "%)" << endl
         << " - Training: " << training_time/1000 << " s"
            << " (" << (training_time/total_time)*100 << "%)" << endl
         << " - Output: " << output_time/1000 << " s"
            << " (" << (output_time/total_time)*100 << "%)";
    cout << endl;
    cout << "====================================================" << endl;

    return 0;
}