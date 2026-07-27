#include <bits/stdc++.h>
using namespace std;

bool flag = 1;
string padZero(int value) {
    return (value < 10 ? "0" : "") + to_string(value);
}

string getCurrentDateTime() {
    time_t now = time(nullptr);
    tm* localTime = localtime(&now);

    string year = to_string(1900 + localTime->tm_year);
    string month = padZero(1 + localTime->tm_mon);
    string day = padZero(localTime->tm_mday);
    string hour = padZero(localTime->tm_hour);
    string minute = padZero(localTime->tm_min);
    string second = padZero(localTime->tm_sec);

    string dateTime = day + "-" + month + "-" + year + " (" + hour + "-" + minute + "-" + second + ").txt";

    return dateTime;
}


//**************************************************************************************************** 


class dataset {
    public:
    vector<vector<float>> train;
    vector<vector<float>> test;
    vector<vector<float>> val;
    dataset(string train_filename, string test_filename, string val_filename) {
        read_file(train_filename, train);
        read_file(test_filename, test);
        read_file(val_filename, val);
    }

    void read_file(string file_name, vector<vector<float>>& arr) {
        ifstream file(file_name);
        string line;
        getline(file, line);
    
        while(getline(file, line)) {
            stringstream ss(line);
            string cell;
    
            vector<float> pixels;
            while (getline(ss, cell, ',')) {
                pixels.push_back(stof(cell));
            }
            arr.push_back(pixels);
        }
    }
};


//****************************************************************************************************


class neural_network {
    public:
    vector<float> y;
    vector<vector<vector<float>>> w;
    vector<vector<float>> n;     
    vector<vector<float>> b;   
    vector<vector<float>> delC_by_delNin_mem;
    vector<vector<vector<float>>> m;
    vector<vector<vector<float>>> v;
    int t;
    float lr;
    float B1, B2;
    
    neural_network(int num_layers, int sizes[]) {
        lr = 0.0005;
        t = 1;
        B1 = 0.9;
        B2 = 0.999;
        y.resize(sizes[num_layers-1]);
        w.resize(num_layers - 1);
        m.resize(num_layers - 1);
        v.resize(num_layers - 1);
        n.resize(num_layers);
        b.resize(num_layers);
        delC_by_delNin_mem.resize(num_layers);

        for(int k=0 ; k<num_layers ; k++) {
            n[k].resize(sizes[k]);
            if(k > 0) {
                delC_by_delNin_mem[k].resize(sizes[k]);
                b[k].resize(sizes[k]);
                init_bias(b, k);
            }
            if(k<num_layers-1) {
                init_weights_m_v(w[k], m[k], v[k], sizes[k+1], sizes[k]);
            }
        }
    }


    void init_weights_m_v(vector<vector<float>>& w, vector<vector<float>>& m, vector<vector<float>>& v, int x, int y) {
        w.resize(x, vector<float>(y));
        m.resize(x, vector<float>(y, 0));
        v.resize(x, vector<float>(y, 0));

        int fan_in = x;
        int fan_out = y;
        random_device rd;
        mt19937 gen(rd());
        float stddev = sqrt(2.0f / (fan_in + fan_out)); // XAVIER INIT
        normal_distribution<float> dist(0.0f, stddev);
        for(int i=0 ; i<x ; i++) {
            for(int j=0 ; j<y ; j++) {
                w[i][j] = dist(gen);
            }
        }
    }

    void init_bias(vector<vector<float>>& b, int k) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dist(-1.0, 1.0);
        for(int i=0 ; i<b[k].size() ; i++) {
            //b[k][i] = 0.0f;
            b[k][i] = dist(gen);
        }
    }


    float activation_tanh(float X) {
        return tanh(X);
    }
    float del_activation_tanh(float X) {
        return 1 - X*X;
    }
    float activation_sigmoid(float X) {
        return 1 / (1 + exp(-X));
    }
    float del_activation_sigmoid(float X) {
        return X * (1 - X);
    }
    float activation_ReLU(float X) {
        if(X > 0) 
            return X;
        else
            return 0.00001f*X;
    }
    float del_activation_ReLU(float X) {
        if(X > 0)
            return 1.0f;
        else
            return 0.00001f;
    }

    void forward_pass() {
        // w[0]: n[0->1], w[1]: n[1->2] ...
        for(int k=0 ; k<w.size() ; k++) {
            for(int i=0 ; i<n[k+1].size() ; i++) {
                n[k+1][i] = 0;
                for(int j=0 ; j<n[k].size() ; j++) {
                    n[k+1][i] += n[k][j]*w[k][i][j];
                }
                n[k+1][i] += b[k+1][i];
                n[k+1][i] = activation_sigmoid(n[k+1][i]);
            }
        }
    }
 
    void backprop() { // Recursion vala.
        for(int k=w.size()-1 ; k>=0 ; k--) {
            for(int i=0 ; i<n[k+1].size() ; i++) {
                for(int j=0 ; j<n[k].size() ; j++) {
                    float delC_by_delW = delC_by_delNin(k+1, i) * n[k][j];
                    w[k][i][j] -= delC_by_delW * lr;
                }
            }
        }
    }
    float delC_by_delNin(int k, int i) {
        if(k == n.size() - 1) {
            return (n[k][i] - y[i]) * del_activation_sigmoid(n[k][i]);
        }
        float delC_by_delNout = 0;
        for(int l=0 ; l<n[k+1].size() ; l++) {
            delC_by_delNout += delC_by_delNin(k+1, l) * w[k][l][i];
        }
        return delC_by_delNout * del_activation_sigmoid(n[k][i]);
    }

    void backprop1() { // Tabulation vala.
        float correction1 = 1 - pow(B1, t);
        float correction2 = 1 - pow(B2, t);
        for(int k=w.size()-1 ; k>=0 ; k--) {
            for(int i=0 ; i<n[k+1].size() ; i++) {
                if(k+1 == n.size() - 1)
                    delC_by_delNin_mem[k+1][i] = (n[k+1][i] - y[i]) * del_activation_sigmoid(n[k+1][i]);
                else {
                    float delC_by_delNout = 0;
                    for(int l=0 ; l<n[k+2].size() ; l++) {
                        delC_by_delNout += delC_by_delNin_mem[k+2][l] * w[k+1][l][i];
                    }
                    delC_by_delNin_mem[k+1][i] = delC_by_delNout * del_activation_sigmoid(n[k+1][i]);
                }
                float delC_by_delb = delC_by_delNin_mem[k+1][i];
                b[k+1][i] -= delC_by_delb * lr; 
                for(int j=0 ; j<n[k].size() ; j++) {
                    float delC_by_delW = delC_by_delNin_mem[k+1][i] * n[k][j];
                    //w[k][i][j] -= delC_by_delW * lr;    // updating weight w(k, i, j)
                    
                    m[k][i][j] = B1 * m[k][i][j] + (1 - B1) * delC_by_delW;
                    v[k][i][j] = B2 * v[k][i][j] + (1 - B2) * delC_by_delW * delC_by_delW;

                    float m_hat = m[k][i][j] / correction1;
                    float v_hat = v[k][i][j] / correction2;

                    w[k][i][j] = w[k][i][j] - lr * (m_hat / (sqrt(v_hat) + 1e-8));
                }
            }
        }
        t++;
    }
    
    void feed_data(vector<vector<float>>& data, int i) {
        for(int j=1 ; j<data[i].size() ; j++) {
            n[0][j-1] = data[i][j]/255.0f;
        }
        for(int j=0 ; j<y.size() ; j++) {
            y[j] = (j == data[i][0]);
        }
    }
    
    int predict() {
        vector<float>& out = n.back();
        return max_element(out.begin(), out.end()) - out.begin();
    }

    void softmax() {
        vector<float>& output = n[n.size()-1];  
        float max_val = *max_element(output.begin(), output.end());
        float sum = 0;
        for (float& val : output) {
            val = exp(val - max_val);
            sum += val;
        }
        for (float& val : output) {
            val /= sum;
        }
    }

    void dropout(float drop_frac) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dist(0, 1.0);

        int tot_size = 0;
        for(int k=1 ; k<n.size()-1 ; k++) 
            tot_size += n[k].size();

        for(int k=1 ; k<n.size()-1 ; k++) {
            unordered_set<int> rand_neurons_set;
            int x = (int)(round)(drop_frac * n[k].size());
            while(rand_neurons_set.size() < x) {
                int rand_ind = floor(dist(gen)*n[k].size());
                if(rand_neurons_set.count(rand_ind) == 0) {
                    rand_neurons_set.insert(rand_ind);
                }
            }
            for(int ind : rand_neurons_set) {
                n[k][ind] = 0;
            }
        }
    }

    float error_meanSq() { // mean square
        float res = 0;
        for(int i=0 ; i<10 ; i++) {
            res += pow(y[i] - n[n.size()-1][i], 2);
        }
        return res/(2*n[n.size()-1].size());
    }

    /* float error_CE() { // cross entropy
        vector<float>& out = n.back();
        float res = 0.0f;
        for (int i = 0; i < out.size(); i++) {
            res -= y[i] * logf(out[i] + 1e-8f);
        }
        return res / logf(2);  
    } */

    void display_confusionMatrix(int confusion[][10]) {
        for(int i=0 ; i<10 ; i++) {
            for(int j=0 ; j<10 ; j++)
                cout<<confusion[i][j]<<"\t";
            cout<<endl;
        }
    }

    void train(dataset data, int epochs) {
        int freq1[10] = {0};
        for(int i=0 ; i<data.train.size() ; i++) {
            freq1[(int)data.train[i][0]]++;
        }
        for(int i=0 ; i<10 ; i++) {
            cout<<i<<" : "<<"Frequency = "<<freq1[i]<<endl;
        }
        cout<<endl;
        
        ofstream outfile("data.txt");  

        float prev_AvgLoss_val = INFINITY;
        for(int epoch=1 ; epoch<=epochs ; epoch++) {
            float AvgLoss_train = 0;
            for(int i=0 ; i<data.train.size() ; i++) {
                if(i % 100 == 0) 
                printf("\r%.3lf %% complete.", ((i+1)*100)/(float)data.train.size());

                feed_data(data.train, i);
                forward_pass();
                //dropout(0.3);
                //softmax();
                backprop1();
                AvgLoss_train += error_meanSq();
            }
            printf("\r");
            cout<<"Epoch: "<<epoch<<". ";
            cout<<"Average Training Loss = "<<AvgLoss_train/data.train.size()<<".";
            outfile<<epoch<<" "<<AvgLoss_train/data.train.size()<<" ";
            
            float AvgLoss_val = 0;
            int tot_correct = 0;
            for(int i=0 ; i<data.val.size() ; i++) {
                feed_data(data.val, i);
                forward_pass();
                softmax();
                if(predict() == (int)data.val[i][0])
                    tot_correct++;

                AvgLoss_val += error_meanSq();
            }
            cout<<" Average Validation Loss = "<<AvgLoss_val/data.val.size()<<". Validation Accuracy = "<<tot_correct*100 / (float)data.val.size()<<" %."<<endl;
            outfile<<AvgLoss_val/data.val.size()<<endl;
            /* if(AvgLoss_val/data.val.size() >= prev_AvgLoss_val) {
                cout<<"AVERAGE VALIDATION LOSS INCREASED. TERMINATING TRAINING."<<endl;
                break;
            }
            else
                prev_AvgLoss_val = AvgLoss_val/data.val.size(); */
        }
        outfile.close();
    
        cout<<"Training complete."<<endl<<endl;
    }

    void test(dataset data) {
        int correct[10] = {0}, incorrect[10] = {0}, freq2[10] = {0};
        float correct_confidence[10] = {0}, incorrect_confidence[10] = {0};
        int confusion[10][10] = {0};
        int tot_correct = 0, tot_incorrect = 0;
        
        
        for(int i=0 ; i<data.test.size() ; i++) {
            feed_data(data.test, i);
            forward_pass();
            softmax();

            int predicted = predict();
            if(predicted == (int)data.test[i][0]) {
                correct[(int)data.test[i][0]]++;
                correct_confidence[(int)data.test[i][0]] += n[n.size()-1][predicted];
                tot_correct++;
            }
            else {
                incorrect[(int)data.test[i][0]]++;
                incorrect_confidence[(int)data.test[i][0]] += n[n.size()-1][predicted];
                tot_incorrect++;
            }
            
            freq2[(int)data.test[i][0]]++;
            confusion[(int)data.test[i][0]][predicted]++;
        }
        
        for(int i=0 ; i<10 ; i++)
            cout<<i<<" : "<<"Frequency = "<<freq2[i]<<". Correctly predicted = "<<correct[i]<<". Incorrectly predicted = "<<incorrect[i]<<". Accuracy = "<<(correct[i] * 100) / (float)(freq2[i])<<endl<<". Confidence when predicted correctly = "<<(correct_confidence[i] * 100) / (float)(correct[i])<<". Confidence when predicted incorrectly = "<<(incorrect_confidence[i] * 100) / (float)(incorrect[i])<<endl<<endl;
        cout<<endl;

        cout<<"Overall Accuracy = "<<(tot_correct * 100) / (float)(tot_correct + tot_incorrect)<<endl<<endl;
        display_confusionMatrix(confusion);
    }

    string save_model() {
        string folder_name = "Saved models";
        filesystem :: create_directory(folder_name);
        string file_name = folder_name + "/" + getCurrentDateTime();
        cout<<"Saving Model to file '"<<file_name<<"'"<<endl;
        ofstream outfile(file_name);

        if (!outfile) {
            cerr << "Error: Could not create file!" << endl;
            return "";
        }

        outfile<<n.size()<<endl<<endl;
        for(int k=0 ; k<n.size() ; k++) {
            outfile<<n[k].size()<<" ";
        }
        outfile<<endl<<endl;

        outfile<<lr<<endl<<endl;

        // save n
        for(int k=1 ; k<n.size()-1 ; k++) {
            for(int i=0 ; i<n[k].size() ; i++) {
                outfile<<n[k][i]<<" ";
            }
            outfile<<endl;
        }
        outfile<<endl;

        // save b
        for(int k=1 ; k<b.size() ; k++) {
            for(int i=0 ; i<b[k].size() ; i++) {
                outfile<<b[k][i]<<" ";
            }
            outfile<<endl;
        }
        outfile<<endl;

        // save w
        for(int k=0 ; k<w.size() ; k++) {
            for(int i=0 ; i<n[k+1].size() ; i++) {
                for(int j=0 ; j<n[k].size() ; j++) {
                    outfile<<w[k][i][j]<<" ";
                }
                outfile<<endl;
            }
            outfile<<endl;
        }

        cout<<"Model saved at: '"<<file_name<<"'"<<endl;
        outfile.close();

        return file_name;
    }
    
    void load_model(string file_name) {
        cout<<endl<<"LOADING..."<<endl<<endl;
        ifstream file(file_name);
        string line;

        getline(file, line);
        int N = stoi(line);
        n.resize(N);
        b.resize(N);
        w.resize(N-1);

        getline(file, line);

        getline(file, line);
        stringstream ss(line);
        string cell;
        for(int k=0 ; k<N ; k++) {
            getline(ss, cell, ' ');
            n[k].resize(stoi(cell));
            if(k > 0) {
                b[k].resize(stoi(cell));
                w[k-1].resize(stoi(cell), vector<float>(n[k-1].size()));
            }
        }
        getline(file, line);

        getline(file, line);
        lr = stof(line);
        cout<<line<<endl;

        getline(file, line);

        // load n
        for(int k=1 ; k<n.size()-1 ; k++) {
            getline(file, line);
            stringstream ss(line);
            string cell;
            for(int i=0 ; i<n[k].size() ; i++) {
                getline(ss, cell, ' ');
                n[k][i] = stof(cell);
            }
        }

        getline(file, line);

        // load b
        for(int k=1 ; k<b.size() ; k++) {
            getline(file, line);
            stringstream ss(line);
            string cell;
            for(int i=0 ; i<b[k].size() ; i++) {
                getline(ss, cell, ' ');
                b[k][i] = stof(cell);
            }
        }

        getline(file, line);

        for(int k=0 ; k<w.size() ; k++) {
            for(int i=0 ; i<n[k+1].size() ; i++) {
                getline(file, line);
                stringstream ss(line);
                string cell;
                for(int j=0 ; j<n[k].size() ; j++) {
                    getline(ss, cell, ' ');
                    w[k][i][j] = stof(cell);
                }
            }
            getline(file, line);
        }
    }
};


// ****************************************************************************************************


int main() {
    int sizes[] = {784, 16, 16, 10};
    neural_network nn(sizeof(sizes)/sizeof(sizes[0]), sizes);
    clock_t start, end;

    cout<<"Loading Datasets..."<<endl;
    dataset data("mnist_train.csv", "mnist_test.csv", "val.csv");
    cout<<"Datasets' Loading Complete."<<endl<<endl;
    

    start = clock();
    cout<<"*************************** TRAINING *************************************"<<endl;
    nn.train(data, 3);
    end = clock();
    cout<<"Time taken for training : "<<(end - start) / (float)CLOCKS_PER_SEC<<" seconds."<<endl<<endl;
    cout<<"*************************** TESTING *************************************"<<endl;
    nn.test(data);

    /* string file_name = nn.save_model();

    nn.load_model(file_name);
    nn.test(data); */
    return 1;
}
