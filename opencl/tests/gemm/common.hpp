namespace x {

// shared between host/device
// xe1 is 8, xe2 is 16
const int SG_SIZE = 8;
const int BLOCK_REG_N = SG_SIZE;
// sg blocking
const int BLOCK_SG_M = 32;
const int BLOCK_SG_N = 16;
const int SG_M = 4;
const int SG_N = 8;
const int BLOCK_WG_K = 64;	// same in sg

};

namespace v2 {

// shared between host/device
// xe1 is 8, xe2 is 16
const int SG_SIZE = 8;
const int BLOCK_REG_N = SG_SIZE;
// sg blocking
const int BLOCK_SG_M = 64;
const int BLOCK_SG_N = 16;
const int SG_M = 2;
const int SG_N = 16;
const int BLOCK_WG_K = 64;	// same in sg

};