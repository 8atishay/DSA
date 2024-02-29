#include <bits/stdc++.h>
#include <iostream>
using namespace std;

struct comp{
    bool operator()(int a, int b){
        return a>b;
    }
};

void printVect(vector<int> v, string title){
    cout<<endl<<title<<endl;
    for(int x: v) cout<<x<<" ";
    cout<<endl;
}

void printPQ(priority_queue<int> pq, string title){
    cout<<endl<<title<<endl;
    while(!pq.empty()){
        cout<<pq.top()<<" ";
        pq.pop();
    }
    cout<<endl;
}

void operate1(vector<int> V){
    vector<int> v = V;
    sort(v.begin(), v.end());
    printVect(v, "V1");
}

void operate2(vector<int> V){
    vector<int> v = V;
    sort(v.begin(), v.end(), comp());
    printVect(v, "V2");
}

void operate3(vector<int> V){
    vector<int> v = V;
    sort(v.begin(), v.end(), [](int a, int b){return a>b;});
    printVect(v, "V2");
}

void operate4(vector<int> V){
    vector<int> v = V;
    priority_queue<int> pq(v.begin(), v.end());
    printPQ(pq, "PQ1");
}

void operate5(vector<int> V){
    vector<int> v = V;
    priority_queue<int, vector<int>, comp> pq;
    for(int x:v) pq.push(x);
    cout<<endl<<"PQ2"<<endl;
    while(!pq.empty()){
        cout<<pq.top()<<" ";
        pq.pop();
    }
    cout<<endl;
}

int main() {
    
    vector<int> v;
    for(int i=0; i<10; i++) v.push_back(rand()%100);
    printVect(v, "V");
    
    operate1(v);
    operate2(v);
    operate3(v);
    operate4(v);
    operate5(v);

    return 0;
}


/* 
Output

V
83 86 77 15 93 35 86 92 49 21 

V1
15 21 35 49 77 83 86 86 92 93 

V2
93 92 86 86 83 77 49 35 21 15 

V2
93 92 86 86 83 77 49 35 21 15 

PQ1
93 92 86 86 83 77 49 35 21 15 

PQ2
15 21 35 49 77 83 86 86 92 93 

*/