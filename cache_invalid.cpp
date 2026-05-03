#include<iostream>
#include<omp.h>
#include<vector>

int main() {
    // int a[10000000];
    std::vector<int> a(1000000000, 0);
    #pragma omp parallel for schedule(static, 4096) shared(a)
    for (size_t i = 0; i < 1000000000; i++) {
        a[i] = a[i] + 1;
    }
}