#include <bits/stdc++.h>
using namespace std;
#define lr 0.1

void init_weights1(float w1[][784]) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(-1.0, 1.0);
    for(int i=0 ; i<16 ; i++) {
        for(int j=0 ; j<784 ; j++) {

            w1[i][j] = dist(gen);

            //cout<<arr[i][j]<<" ";
        }
        //cout<<endl;
    }
}
    
void init_weights2(float w2[][16]) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(-1.0, 1.0);
    for(int i=0 ; i<16 ; i++) {
        for(int j=0 ; j<16 ; j++) {

            w2[i][j] = dist(gen);

            //cout<<arr[i][j]<<" ";
        }
        //cout<<endl;
    }
}
void init_weights3(float w3[][16]) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(-1.0, 1.0);
    for(int i=0 ; i<10 ; i++) {
        for(int j=0 ; j<16 ; j++) {

            w3[i][j] = dist(gen);

            //cout<<arr[i][j]<<" ";
        }
        //cout<<endl;
    }
}

float activate_sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

void forward_1(float x[], float n1[], float w1[][784]) {
    for(int i=0 ; i<16 ; i++) {
        n1[i] = 0;
        for(int j=0 ; j<784 ; j++) {
            n1[i] += x[j]*w1[i][j]; 
        }
        n1[i] = activate_sigmoid(n1[i]);
    }
}
void forward_2(float n1[], float n2[], float w2[][16]) {
    for(int i=0 ; i<16 ; i++) {
        n2[i] = 0;
        for(int j=0 ; j<16 ; j++) {
            n2[i] += n1[j]*w2[i][j]; 
        }
        n2[i] = activate_sigmoid(n2[i]);
    }
}
void forward_3(float n2[], float out[], float w3[][16]) {
    for(int i=0 ; i<10 ; i++) {
        out[i] = 0;
        for(int j=0 ; j<16 ; j++) {
            out[i] += n2[j]*w3[i][j]; 
        }
        out[i] = activate_sigmoid(out[i]);
    }
}

float error(float out[], float y[]) {
    float res = 0;
    for(int i=0 ; i<10 ; i++) {
        res += pow(y[i] - out[i], 2);
    }
    return res/20;
}

void backprop_w3(float w3[][16], float out[], float y[], float n2[]) {
    for(int i=0 ; i<10 ; i++) {
        for(int j=0 ; j<16 ; j++) {
            float grad = (out[i] - y[i]) * out[i] * (1 - out[i]) * n2[j];
            w3[i][j] -= grad * lr;
        }
    }
}

void backprop_w2(float w2[][16], float out[], float y[], float n1[], float n2[], float w3[][16]) {
    
    for(int j=0 ; j<16 ; j++) {
        for(int k=0 ; k<16 ; k++) {
            float grad1 = 0, grad2 = n2[k] * (1-n2[k]) * n1[j];
            for(int i=0 ; i<10 ; i++) {
                grad1 += (out[i]-y[i]) * out[i] * (1-out[i]) * w3[i][k];
            }
            w2[k][j] -= grad1 * grad2 * lr;
        }
    }
}

void backprop_w1(float w1[][784], float w2[][16], float w3[][16], float x[], float n1[], float n2[], float out[], float y[]) {
    for(int l=0 ; l<784 ; l++) {
        for(int j=0 ; j<16 ; j++) {
            float grad2 = 0, grad3 = n1[j] * (1-n1[j]) * x[l];
            for(int k=0 ; k<16 ; k++) {
                float grad1 = 0;
                for(int i=0 ; i<10 ; i++) {
                    grad1 += (out[i]-y[i]) * out[i] * (1-out[i]) * w3[i][k];
                }
                grad2 += grad1 * n2[k] * (1 - n2[k]) * w2[k][j];
            }
            w1[j][l] -= grad2 * grad3 * lr;
        }
    }
}

void softmax(float out[]) {
    float max_val = INT_MIN;
    for(int i=0 ; i<10 ; i++) {
        if(out[i] > max_val)
            max_val = out[i];
    }
    float sum = 0;
    for(int i=0 ; i<10 ; i++) {
        out[i] = exp(out[i] - max_val);
        sum += out[i];
    }
    for(int i=0 ; i<10 ; i++) {
        out[i] /= sum;
    }
}

int predict(float out[]) {
    int max_val = 0;
    for(int i=1 ; i<10 ; i++) {
        if(out[i] > out[max_val])
            max_val = i;
    }
    return max_val;
}

void read_images(vector<vector<float>>& train_images, float x[], float y[], int i) {
    for(int j=1 ; j<=784 ; j++) {
        x[j] = train_images[i][j]/255;
    }
    for(int j=0 ; j<10 ; j++) {
        y[j] = (j == train_images[i][0]);
    }
}

void read_file(vector<vector<float>>& train_images, string file_name) {
    ifstream file(file_name);
    string line;
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;

        vector<float> pixels;
        while (getline(ss, cell, ',')) {
            pixels.push_back(stof(cell));
        }
        train_images.push_back(pixels);
    }
}

void display_confusionMatrix(int confusion[][10]) {
    for(int i=0 ; i<10 ; i++) {
        for(int j=0 ; j<10 ; j++)
            cout<<confusion[i][j]<<" ";
        cout<<endl;
    }
}

int main() {
    
    //trainnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
    vector<vector<float>> train_images;
    read_file(train_images, "train1.csv");
    clock_t start, end;

    int freq1[10] = {0};
    for(int i=0 ; i<train_images.size() ; i++) {
        freq1[(int)train_images[i][0]]++;
    }
    for(int i=0 ; i<10 ; i++)
        cout<<i<<" : "<<"Frequency = "<<freq1[i]<<endl;
    cout<<endl;
    
    float w1[16][784];
    init_weights1(w1);
    float w2[16][16];
    init_weights2(w2);
    float w3[10][16];
    init_weights3(w3);
    
    float x[784];
    float n1[16];
    float n2[16];
    float out[10];

    float y[10];

    ofstream outfile("data.txt");  

    start = clock();
    for(int epoch=1 ; epoch<=2 ; epoch++) {
        cout<<"Epoch: "<<epoch<<". ";
        float AvgLoss = 0;
        for(int i=0 ; i<train_images.size() ; i++) {
            read_images(train_images, x, y, i);

            forward_1(x, n1, w1);
            forward_2(n1, n2, w2);
            forward_3(n2, out, w3);
            
            backprop_w3(w3, out, y, n2);
            backprop_w2(w2, out, y, n1, n2, w3);
            backprop_w1(w1, w2, w3, x, n1, n2, out, y);

            AvgLoss += error(out, y);
        }
        cout<<"Average Loss = "<<AvgLoss/train_images.size()<<endl;
        outfile<<epoch<<" "<<AvgLoss/train_images.size()<<endl;
    }
    outfile.close();
    end = clock();
    cout<<"Time taken for training : "<<(end - start) / (float)CLOCKS_PER_SEC<<" seconds."<<endl<<endl;

    cout<<"Training complete."<<endl<<endl;

    //testttttttttttttttttttttttttttttttttttttttttttttttt
    vector<vector<float>> test_images;
    read_file(test_images, "test1.csv");

    int correct[10] = {0}, incorrect[10] = {0}, freq2[10] = {0};
    float correct_confidence[10] = {0}, incorrect_confidence[10] = {0};
    int confusion[10][10] = {0};


    for(int i=0 ; i<test_images.size() ; i++) {
        read_images(test_images, x, y, i);

        forward_1(x, n1, w1);
        forward_2(n1, n2, w2);
        forward_3(n2, out, w3);

        softmax(out);

        int predicted = predict(out);
        if(predicted == (int)test_images[i][0]) {
            correct[(int)test_images[i][0]]++;
            correct_confidence[(int)test_images[i][0]] += out[predicted];
        }
        else {
            incorrect[(int)test_images[i][0]]++;
            incorrect_confidence[(int)test_images[i][0]] += out[predicted];
        }
        
        freq2[(int)test_images[i][0]]++;
        confusion[(int)test_images[i][0]][predicted]++;
    }

    for(int i=0 ; i<10 ; i++)
        cout<<i<<" : "<<"Frequency = "<<freq2[i]<<". Correctly predicted = "<<correct[i]<<". Incorrectly predicted = "<<incorrect[i]<<". Accuracy = "<<(correct[i] * 100) / (float)(correct[i] + incorrect[i])<<endl<<". Confidence when predicted correctly = "<<(correct_confidence[i] * 100) / (float)(correct[i])<<". Confidence when predicted incorrectly = "<<(incorrect_confidence[i] * 100) / (float)(incorrect[i])<<endl<<endl;
    cout<<endl;

    int tot_correct = 0, tot_incorrect = 0;
    for(int i=0 ; i<10 ; i++) {
        tot_correct += correct[i];
        tot_incorrect += incorrect[i];
    }
    cout<<"Overall Accuracy = "<<(tot_correct * 100) / (float)(tot_correct + tot_incorrect)<<endl<<endl;

    display_confusionMatrix(confusion);

    float sum = 0;
    for(int i=0 ; i<10 ; i++) {
        sum += out[i];
        cout<<out[i]<<endl;
    }
    cout<<sum<<endl;
    return 0;
}
