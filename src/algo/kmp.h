/**
* This is an implementation of the Knuth Morris Pratt algorithm
* TODO: more comments and docs
*/
int kmp(const char * pattern, const char * text, int * output);

/**
* Build border array for text
*/
void border(const char * text, int * border);
