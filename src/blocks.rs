use cgmath::*;
use vulkano as vk;

#[derive(Copy, Clone, Default)]
pub struct Vertex {
    position: [f32; 3],
    color   : [f32; 3],
    tex_coords: [f32; 2],
    light_intensity: f32
}

impl Vertex {
    pub fn new(pos: [f32; 3], color: [f32; 3], tex: [f32; 2], light: f32) -> Self {
        Vertex {
            position: pos, color: color, tex_coords: tex, light_intensity: light
        }
    }
}

vk::impl_vertex!(Vertex, position, color, tex_coords, light_intensity);

#[derive(Copy, Clone)]
pub enum BlockType {
    Air,
    Grass,
    Sand,
    Stone
}

const BLOCKSIZE: f32 = 0.03;
const CHUNK_SIZE_Y: usize = 256; //"layer"
const CHUNK_SIZE_Z: usize = 16; //z-axis
const CHUNK_SIZE_X: usize = 16; //x-axis

const CHUNK_MAXIDX_Y: usize = CHUNK_SIZE_Y - 1; //"layer"
const CHUNK_MAXIDX_Z: usize = CHUNK_SIZE_Z - 1; //z-axis
const CHUNK_MAXIDX_X: usize = CHUNK_SIZE_X - 1; //x-axis

struct TextureQuad  {
    top_left: [f32; 2],
    top_right: [f32; 2],
    bottom_left: [f32; 2],
    bottom_right: [f32; 2],
}
const QUADSIZE: f32 = 1.0 / 8.0;
const GRASS: TextureQuad = TextureQuad { 
    top_left: [0.0, 0.0],
    top_right: [QUADSIZE, 0.0],
    bottom_left: [0.0, QUADSIZE],
    bottom_right: [QUADSIZE, QUADSIZE]
};
const GRASS2: TextureQuad = TextureQuad { 
    top_left: [0.0, QUADSIZE],
    top_right: [QUADSIZE, QUADSIZE],
    bottom_left: [0.0, QUADSIZE * 2.0],
    bottom_right: [QUADSIZE, QUADSIZE * 2.0]
};
const GRASS3: TextureQuad = TextureQuad { 
    top_left: [0.0, QUADSIZE * 2.0],
    top_right: [QUADSIZE, QUADSIZE * 2.0],
    bottom_left: [0.0, QUADSIZE * 3.0],
    bottom_right: [QUADSIZE, QUADSIZE * 3.0]
};
const GRASS4: TextureQuad = TextureQuad { 
    top_left: [0.0, QUADSIZE * 3.0],
    top_right: [QUADSIZE, QUADSIZE * 3.0],
    bottom_left: [0.0, QUADSIZE * 4.0],
    bottom_right: [QUADSIZE, QUADSIZE * 4.0]
};
const GRASS5: TextureQuad = TextureQuad { 
    top_left: [0.0, QUADSIZE * 4.0],
    top_right: [QUADSIZE, QUADSIZE * 4.0],
    bottom_left: [0.0, QUADSIZE * 5.0],
    bottom_right: [QUADSIZE, QUADSIZE * 5.0]
};
const GRASS6: TextureQuad = TextureQuad { 
    top_left: [0.0, QUADSIZE * 5.0],
    top_right: [QUADSIZE, QUADSIZE * 5.0],
    bottom_left: [0.0, QUADSIZE * 6.0],
    bottom_right: [QUADSIZE, QUADSIZE * 6.0]
};
const GRASS7: TextureQuad = TextureQuad { 
    top_left: [0.0, QUADSIZE * 6.0],
    top_right: [QUADSIZE, QUADSIZE * 6.0],
    bottom_left: [0.0, QUADSIZE * 7.0],
    bottom_right: [QUADSIZE, QUADSIZE * 7.0]
};
const GRASS8: TextureQuad = TextureQuad { 
    top_left: [0.0, QUADSIZE * 7.0],
    top_right: [QUADSIZE, QUADSIZE * 7.0],
    bottom_left: [0.0, QUADSIZE * 8.0],
    bottom_right: [QUADSIZE, QUADSIZE * 8.0]
};


const GRASSES: [TextureQuad; 8] = [GRASS, GRASS2, GRASS3, GRASS4, GRASS5, GRASS6, GRASS7, GRASS8];

const GRASS_DIRT: TextureQuad = TextureQuad { 
    top_left: [QUADSIZE, 0.0],
    top_right: [QUADSIZE * 2.0, 0.0],
    bottom_left: [QUADSIZE, QUADSIZE],
    bottom_right: [QUADSIZE * 2.0, QUADSIZE]
};
const DIRT: TextureQuad = TextureQuad { 
    top_left: [QUADSIZE * 2.0, 0.0],
    top_right: [QUADSIZE * 3.0, 0.0],
    bottom_left: [QUADSIZE * 2.0, QUADSIZE],
    bottom_right: [QUADSIZE * 3.0, QUADSIZE]
};


pub struct Chunk {
    //these are the "world space" coordinates of the chunk, but they are not the coordinates of blocks.
    //these coordinates go like (0,0), (0,1),(0,2), (1,0), (1,1), ... 
    //and these coordinates follow opengl/vulkan X and Z standard: +X is right, -Z is far
    //always i32 numbers
    x: i32,
    z: i32,
    blocks: [[[BlockType; CHUNK_SIZE_X]; CHUNK_SIZE_Z]; CHUNK_SIZE_Y]
}

pub struct World {
    //chunks are vec of 3d block array
    //index in array is not meaningful
    //To find a chunk, must iterate. Usally there's only a small number of loaded chunks.
    chunks: Vec<Chunk> 
}

impl World {

    fn get_chunk(&self, x: i32, z: i32) -> Option<&Chunk> {
        let mut found = None;
        for chunk in self.chunks.iter() {
            if chunk.x == x && chunk.z == z {
                found = Some(chunk)
            }
        }
        return found
    }

    fn generate_block(chunk_x: i32, chunk_z: i32, x: usize, y: usize, z: usize, last_index: u32, color: [f32; 3]) -> (Vec<Vertex>, Vec<u32>) {
        use std::collections::hash_map::DefaultHasher;
        let mut s = DefaultHasher::new();
        use std::hash::*;
        use std::hash::Hash;
        #[derive(Hash)]
        struct HashedCoords {
            chunk_x: i32, chunk_z: i32, x: usize, y: usize, z: usize
        }
        let hashed = HashedCoords { chunk_x, chunk_z, x, y, z};
        hashed.hash(&mut s);
        let hashval = s.finish();
        let x = (x as f32 * BLOCKSIZE) + ((chunk_x * CHUNK_SIZE_X as i32) as f32 * BLOCKSIZE);
        let y = y as f32 * BLOCKSIZE;
        let z = (z as f32 * BLOCKSIZE) - ((chunk_z * CHUNK_SIZE_Z as i32) as f32 * BLOCKSIZE);

        let blk_1x = 1.0 * BLOCKSIZE;
        let blk_2x = 2.0 * BLOCKSIZE;
        let back = -1.0 * BLOCKSIZE;
        let red: [f32; 3] = [1.0, 0.0, 0.0];
        let green: [f32; 3] = [0.0, 1.0, 0.0];
        let mut vertices = vec![];
        let mut indices: Vec<u32> = vec![];

        let front_top_left      = [blk_1x + x, blk_1x + y, -z];
        let front_top_right     = [blk_2x + x, blk_1x + y, -z];
        let front_bottom_right  = [blk_2x + x, y,          -z];
        let front_bottom_left   = [blk_1x + x, y,          -z];

        let back_top_left      = [blk_1x + x, blk_1x + y, back - z];
        let back_top_right     = [blk_2x + x, blk_1x + y, back - z];
        let back_bottom_right  = [blk_2x + x, y,          back - z];
        let back_bottom_left   = [blk_1x + x, y,          back - z];

        let mut v = |pos: [f32; 3], color: [f32; 3], tex: [f32; 2], light_intensity: f32| -> u32 {
            vertices.push(Vertex::new(pos, color, tex, light_intensity));
            return vertices.len() as u32 - 1
        };
       
        let modulus = hashval % GRASSES.len() as u64;
        println!("Hash: {}, {}", hashval, modulus);        
        let grass_texture = &GRASSES[modulus as usize];


        //front face
        {
            let a = v(front_top_left, green, GRASS_DIRT.top_left, 0.6);
            let b = v(front_top_right, green, GRASS_DIRT.top_right, 0.6);
            let c = v(front_bottom_right, green, GRASS_DIRT.bottom_right, 0.6);
            let d = v(front_bottom_left, green, GRASS_DIRT.bottom_left, 0.6);
            indices.extend(&[a, b, c, c, d, a]);
        }
        //right face
        {
            let a = v(front_top_right, red, GRASS_DIRT.top_left, 0.6);
            let b = v(back_top_right, red, GRASS_DIRT.top_right, 0.6);
            let c = v(back_bottom_right, red, GRASS_DIRT.bottom_right, 0.6);
            let d = v(front_bottom_right, red, GRASS_DIRT.bottom_left, 0.6);
            indices.extend(&[a, b, c, c, d, a]);
        }
        //left face
        {
            let a = v(back_top_left, red, GRASS_DIRT.top_left, 0.3);
            let b = v(front_top_left, red, GRASS_DIRT.top_right, 0.3);
            let c = v(front_bottom_left, red, GRASS_DIRT.bottom_right, 0.3);
            let d = v(back_bottom_left, red, GRASS_DIRT.bottom_left, 0.3);
            indices.extend(&[a, b, c, c, d, a]);
        }
        //back face
        {
            let a = v(back_top_right, red, GRASS_DIRT.top_left, 0.3);
            let b = v(back_top_left, red, GRASS_DIRT.top_right, 0.3);
            let c = v(back_bottom_left, red, GRASS_DIRT.bottom_right, 0.3);
            let d = v(back_bottom_right, red, GRASS_DIRT.bottom_left, 0.3);
            indices.extend(&[a, b, c, c, d, a]);
        }
        //top face
        {
            let a = v(back_top_left, red, grass_texture.top_left, 1.0);
            let b = v(back_top_right, red, grass_texture.top_right, 1.0);
            let c = v(front_top_right, red, grass_texture.bottom_right, 1.0);
            let d = v(front_top_left, red, grass_texture.bottom_left, 1.0);
            indices.extend(&[a, b, c, c, d, a]);
        }
        //bottom face
        {
            let a = v(front_bottom_left, red, DIRT.top_left, 0.1);
            let b = v(front_bottom_right, red, DIRT.top_right, 0.1);
            let c = v(back_bottom_right, red, DIRT.bottom_right, 0.1);
            let d = v(back_bottom_left, red, DIRT.bottom_left, 0.1);
            indices.extend(&[a, b, c, c, d, a]);
        }

        let indices: Vec<u32> = indices.iter().map(|x| x + last_index).collect(); 

        return (vertices, indices);
    }

    //x, y and z are relative to the chunk itself!
    //Therefore, if we pass a negative x or x > WIDTH, we need to check another chunk.
    //Same goes for Y and Z.
    //THis code *only* supports checking neighboring chunks.
    //The chunk you pass here is the chunk which x y z are relative to. But you can pass
    //negative numbers or even X Y Z > MAX, therefore we will check a neighboring chunk. 
    //if chunk is unloaded, should return BlockType::Air
    pub fn blocktype(&self, chunk: &Chunk, x: i32, y: i32, z: i32) -> BlockType {
        //edge case: top-layer check: anything above CHUNK_SIZE_Y is air.
        if y > CHUNK_MAXIDX_Y as i32 { return BlockType::Air }
        //edge case: bottom-later check: Anything under 0 is air.
        if y < 0 { return BlockType::Air }
        
        let mut actual_chunk_z = chunk.z; 
        let mut actual_chunk_x = chunk.x; 

        let mut actual_z = z;
        let mut actual_x = x;

        //Chunk edge detection
        //If X and Z are negative or above their limits, then must check neighboring chunk.
        //First detect for X:
        if x < 0 { 
            actual_chunk_x -= 1;
            //suppose we passed -1 as X. Then we need to get the 16th block on the previous
            //chunk. -2 would be 15th and so on.
            actual_x = CHUNK_SIZE_X as i32 + x;
        }
        else if x > CHUNK_MAXIDX_X as i32 { 
            actual_chunk_x += 1; 
            //in this case we need to get the index in the next chunk. 
            //If we passed 16 it would be 0 in the next one.
            actual_x = x - CHUNK_SIZE_X as i32 
        }

        //Detect for Y but in this case the coordinates are flipped: 
        //behind 0 should go to a positive Z chunk, because we are follwing vulkan conventions
        //remember that the indexes in the block array are not the same as worldspace coordinates,
        //but the chunk positions *do* follow worldspace coordinates! 
        if z < 0 { 
            actual_chunk_z += 1;
            //as i'm going in +z, then we must access the closest block in the neighboring chunk 
            //which will be the 16th row (index 15) if I pass -1
            actual_z = CHUNK_SIZE_X as i32 + z;
        }
        else if z > CHUNK_MAXIDX_Z as i32 { 
            actual_chunk_z -= 1;
            actual_z = z - CHUNK_SIZE_Z as i32 
        }

        //are we sill in the same chunk?

        let same_chunk = actual_chunk_x == chunk.x && actual_chunk_z == chunk.z;
        if same_chunk {
            //y z x
            chunk.blocks[y as usize][actual_z as usize][actual_x as usize]
        } else {
            let chunk_find = self.get_chunk(actual_chunk_x, actual_chunk_z);
            match chunk_find {
                None => BlockType::Air, //unloaded chunk, block is air
                Some(actual_chunk) => {
                    actual_chunk.blocks[y as usize][actual_z as usize][actual_x as usize]
                }
            }
        }
    }


    pub fn meshgen(&self) -> (Vec<Vertex>, Vec<u32>) {
        let mut vertices = vec![];
        let mut indices = vec![];

        let offsets_to_check = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ];

        for chunk in self.chunks.iter() {
            for (y, layer) in chunk.blocks.iter().enumerate() {
                for (z, row) in layer.iter().enumerate() {
                    for (x, item) in row.iter().enumerate() {
    
                        
                        if let BlockType::Air = item {
                            continue;
                        }
                        
                        let mut found_air = false;
                        for (offx, offy, offz) in offsets_to_check {
                            let blocktype = self.blocktype(chunk, x as i32 + offx, y as i32 +offy, z as i32 + offz);
                            if let BlockType::Air = blocktype {
                                found_air = true;
                                break;
                            }
                        }
                   
                        if !found_air { continue };

                        let color = match item {
                            BlockType::Sand => [0.5, 0.5, 0.0],
                            BlockType::Grass => [0.0, 0.7, 0.0],
                            BlockType::Stone => [0.3, 0.3, 0.3],
                            _ => {[0.0, 0.0, 1.0]}
                        };

                        let (generated_vertices, generated_indices) = World::generate_block(
                            chunk.x, chunk.z,
                            x, y, z, vertices.len() as u32, color);
                        vertices.extend(generated_vertices);
                        indices.extend(generated_indices);
                    }
                }
            }
        }
        return (vertices, indices);
    }


    pub fn chunkgen(chunk_z: i32, chunk_x: i32) -> Chunk {
        let mut chunk = Chunk {
            x: chunk_x,
            z: chunk_z,
            blocks: [[[BlockType::Air; 16]; 16]; 256]
        };
        chunk.blocks[0][0][0] = BlockType::Grass;
        for y in 0..1 {
            for z in 0..16 {
                for x in 0..16 {
                    if x > 8 {
                        chunk.blocks[y][z][x] = BlockType::Grass;
                    } else {
                        chunk.blocks[y][z][x] = BlockType::Sand;
                    }
                }    
            }    
        }

        //stone core
        for y in 0..20 {
            for z in 6..11 {
                for x in 6..11 {
                    chunk.blocks[y][z][x] = BlockType::Air;
                }    
            }    
        }

        chunk
    }

    pub fn worldgen() -> World {
        
        let mut world = World {
            chunks: vec![]
        };
        
        //x, z chunks around player
        let chunks_to_gen = [
            (0, 0), //player chunk
            /*(0, -1), //clockwise around player
            (0, -2), //clockwise around player
            (0, -3), //clockwise around player
            (0, -4), //clockwise around player
            (0, -5), //clockwise around player
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),*/
        ];

        for (x, z) in chunks_to_gen {
            world.chunks.push(World::chunkgen(z, x));
        }
        return world;
    }
}