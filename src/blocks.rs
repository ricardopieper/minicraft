use bitflags::bitflags;
use cgmath::*;
use noise::{NoiseFn, Perlin, Seedable};

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
    tex_coords: [f32; 2],
    light_intensity: f32,
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
                0 => Float32x3,
                1 => Float32x3,
                2 => Float32x2,
                3 => Float32];

    pub fn new(pos: [f32; 3], color: [f32; 3], tex: [f32; 2], light: f32) -> Self {
        Vertex {
            position: pos,
            color,
            tex_coords: tex,
            light_intensity: light,
        }
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

//vk::impl_vertex!(Vertex, position, color, tex_coords, light_intensity);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BlockType {
    Air,
    Grass,
    Dirt,
    Sand,
    Stone,
}

const BLOCKSIZE: f32 = 0.03;
const CHUNK_SIZE_Y: usize = 256; //"layer"
const CHUNK_SIZE_Z: usize = 16; //z-axis
const CHUNK_SIZE_X: usize = 16; //x-axis

const CHUNK_MAXIDX_Y: usize = CHUNK_SIZE_Y - 1; //"layer"
const CHUNK_MAXIDX_Z: usize = CHUNK_SIZE_Z - 1; //z-axis
const CHUNK_MAXIDX_X: usize = CHUNK_SIZE_X - 1; //x-axis

#[derive(Copy, Clone)]
struct TextureQuad {
    top_left: [f32; 2],
    top_right: [f32; 2],
    bottom_left: [f32; 2],
    bottom_right: [f32; 2],
}

impl TextureQuad {
    const fn new(row: u32, col: u32) -> Self {
        let row = row as f32;
        let col = col as f32;
        Self {
            top_left: [QUADSIZE * col, QUADSIZE * row],
            top_right: [QUADSIZE * (col + 1.0), QUADSIZE * row],
            bottom_left: [QUADSIZE * col, QUADSIZE * (row + 1.0)],
            bottom_right: [QUADSIZE * (col + 1.0), QUADSIZE * (row + 1.0)],
        }
    }

    const fn rotate90(&self) -> Self {
        Self {
            top_left: self.bottom_left,
            top_right: self.top_left,
            bottom_right: self.top_right,
            bottom_left: self.bottom_right,
        }
    }
    const fn rotate180(&self) -> Self {
        self.rotate90().rotate90()
    }
    const fn rotate270(&self) -> Self {
        self.rotate90().rotate90().rotate90()
    }
    const fn all_rotations(&self) -> [TextureQuad; 4] {
        [*self, self.rotate90(), self.rotate180(), self.rotate270()]
    }
}

const QUADSIZE: f32 = 1.0 / 8.0;
const GRASS: TextureQuad = TextureQuad::new(0, 0);
const GRASS_DIRT: TextureQuad = TextureQuad::new(0, 1);
const DIRT: TextureQuad = TextureQuad::new(0, 2);
const SAND: TextureQuad = TextureQuad::new(0, 3);

struct BlockTypeTexture {
    top: [TextureQuad; 4],
    bottom: [TextureQuad; 4],
    left: [TextureQuad; 4],
    right: [TextureQuad; 4],
    front: [TextureQuad; 4],
    back: [TextureQuad; 4],
}

impl BlockTypeTexture {
    const fn uniform_with_rotations(texquad: TextureQuad) -> Self {
        Self {
            top: texquad.all_rotations(),
            bottom: texquad.all_rotations(),
            left: texquad.all_rotations(),
            right: texquad.all_rotations(),
            front: texquad.all_rotations(),
            back: texquad.all_rotations(),
        }
    }

    const fn grassdirt_like(sides: TextureQuad, top: TextureQuad, bottom: TextureQuad) -> Self {
        Self {
            top: top.all_rotations(),
            bottom: bottom.all_rotations(),
            left: [sides, sides, sides, sides],
            right: [sides, sides, sides, sides],
            front: [sides, sides, sides, sides],
            back: [sides, sides, sides, sides],
        }
    }
}

const BLOCK_TEXTURE_GRASS: BlockTypeTexture =
    BlockTypeTexture::grassdirt_like(GRASS_DIRT, GRASS, DIRT);
const BLOCK_TEXTURE_SAND: BlockTypeTexture = BlockTypeTexture::uniform_with_rotations(SAND);
const BLOCK_TEXTURE_DIRT: BlockTypeTexture = BlockTypeTexture::uniform_with_rotations(DIRT);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkPosition {
    x: i32,
    z: i32,
}
pub fn chunkpos(x: i32, z: i32) -> ChunkPosition {
    ChunkPosition { x, z }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BlockPositionInChunk {
    pos: Vector3<usize>,
}
pub fn blockpos_chunk(x: usize, y: usize, z: usize) -> BlockPositionInChunk {
    BlockPositionInChunk { pos: vec3(x, y, z) }
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BlockPositionInWorld {
    pos: Vector3<i32>,
}
pub fn blockpos_world(x: i32, y: i32, z: i32) -> BlockPositionInWorld {
    BlockPositionInWorld { pos: vec3(x, y, z) }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    //these are the "world space" coordinates of the chunk, but they are not the coordinates of blocks.
    //these coordinates go like (0,0), (0,1),(0,2), (1,0), (1,1), ...
    //and these coordinates follow opengl/vulkan X and Z standard: +X is right, -Z is far
    //always i32 numbers
    chunk_pos: ChunkPosition,
    blocks: [[[BlockType; CHUNK_SIZE_X]; CHUNK_SIZE_Z]; CHUNK_SIZE_Y],
}

impl Chunk {
    pub fn new(chunk_pos: ChunkPosition) -> Self {
        let blocks = [[[BlockType::Air; CHUNK_SIZE_X]; CHUNK_SIZE_Z]; CHUNK_SIZE_Y];
        Self {
            chunk_pos: chunk_pos,
            blocks,
        }
    }

    pub fn set_block(&mut self, chunkspace_coords: BlockPositionInChunk, block_type: BlockType) {
        let pos = chunkspace_coords.pos;
        self.blocks[pos.y][pos.z][pos.x] = block_type;
    }

    //given x and z block coordinates in the world, returns the chunk coordinates
    pub fn block_to_chunk_coords(block: BlockPositionInWorld) -> ChunkPosition {
        ChunkPosition {
            x: (block.pos.x as f32 / CHUNK_SIZE_X as f32) as i32,
            z: (block.pos.z as f32 / CHUNK_SIZE_Z as f32) as i32,
        }
    }

    pub fn get_chunk_coords(x: i32, z: i32) -> ChunkPosition {
        ChunkPosition {
            x: (x as f32 / CHUNK_SIZE_X as f32) as i32,
            z: (z as f32 / CHUNK_SIZE_Z as f32) as i32,
        }
    }

    pub fn chunk_starting_block(chunk_pos: ChunkPosition) -> BlockPositionInWorld {
        BlockPositionInWorld {
            pos: vec3(
                chunk_pos.x * CHUNK_SIZE_X as i32,
                0,
                chunk_pos.z * CHUNK_SIZE_Z as i32,
            ),
        }
    }

    pub fn get_chunkspace_coords_from_worldspace_coords(
        chunk_pos: ChunkPosition,
        block_pos: BlockPositionInWorld,
    ) -> BlockPositionInChunk {
        let chunk_starting_block = Chunk::chunk_starting_block(chunk_pos);
        let pos = block_pos.pos - chunk_starting_block.pos;
        return blockpos_chunk(pos.x as usize, pos.y as usize, pos.z as usize);
    }
}

pub struct World {
    //chunks are vec of 3d block array
    //index in array is not meaningful
    chunks: std::collections::HashMap<ChunkPosition, Chunk>,
}

bitflags! {
    struct Faces: u8 {
        const None =   0 << 0;
        const Front =  1 << 0;
        const Right =  1 << 1;
        const Back =   1 << 2;
        const Left =   1 << 3;
        const Top =    1 << 4;
        const Bottom = 1 << 5;
    }
}

const OFFSETS_TO_CHECK: [(Faces, i32, i32, i32); 6] = [
    (Faces::Front, 0, 0, -1),
    (Faces::Right, 1, 0, 0),
    (Faces::Left, -1, 0, 0),
    (Faces::Top, 0, 1, 0),
    (Faces::Bottom, 0, -1, 0),
    (Faces::Back, 0, 0, 1),
];

impl World {
    fn get_chunk(&self, chunk_pos: ChunkPosition) -> Option<&Chunk> {
        self.chunks.get(&chunk_pos)
    }

    fn expect_chunk_mut(&mut self, chunk_pos: ChunkPosition) -> &mut Chunk {
        self.chunks.get_mut(&chunk_pos).unwrap()
    }

    fn generate_block(
        ChunkPosition {
            x: chunk_x,
            z: chunk_z,
        }: ChunkPosition,
        x: usize,
        y: usize,
        z: usize,
        last_index: u32,
        block_texture: BlockTypeTexture,
        faces_to_render: Faces,
    ) -> (Vec<Vertex>, Vec<u32>) {
        let mut s = DefaultHasher::new();
        use std::hash::*;
        #[derive(Hash)]
        struct HashedCoords {
            chunk_x: i32,
            chunk_z: i32,
            x: usize,
            y: usize,
            z: usize,
        }
        let hashed =
            HashedCoords {
                chunk_x,
                chunk_z,
                x,
                y,
                z,
            };
        hashed.hash(&mut s);
        let hashval = s.finish();

        //position in "render space"
        let x = ((chunk_x as f32 * CHUNK_SIZE_X as f32) + x as f32) * BLOCKSIZE as f32;
        let y = y as f32 * BLOCKSIZE;
        let z = ((chunk_z as f32 * CHUNK_SIZE_Z as f32) + z as f32) * BLOCKSIZE as f32;

        let blk_1x = 1.0 * BLOCKSIZE;
        let blk_2x = 2.0 * BLOCKSIZE;
        let back = -1.0 * BLOCKSIZE;
        let red: [f32; 3] = [1.0, 0.0, 0.0];
        let green: [f32; 3] = [0.0, 1.0, 0.0];

        let mut vertices = vec![];
        let mut indices: Vec<u32> = vec![];

        let front_top_left = [blk_1x + x, blk_1x + y, -z];
        let front_top_right = [blk_2x + x, blk_1x + y, -z];
        let front_bottom_right = [blk_2x + x, y, -z];
        let front_bottom_left = [blk_1x + x, y, -z];

        let back_top_left = [blk_1x + x, blk_1x + y, back - z];
        let back_top_right = [blk_2x + x, blk_1x + y, back - z];
        let back_bottom_right = [blk_2x + x, y, back - z];
        let back_bottom_left = [blk_1x + x, y, back - z];

        let mut v = |pos: [f32; 3], color: [f32; 3], tex: [f32; 2], light_intensity: f32| -> u32 {
            vertices.push(Vertex::new(pos, color, tex, light_intensity));
            return vertices.len() as u32 - 1;
        };

        let texture_variation = hashval % 4;
        //clockwise winding order
        //front face
        if faces_to_render.contains(Faces::Front) {
            let tex = block_texture.front[texture_variation as usize];
            let a = v(front_top_left, green, tex.top_left, 0.6);
            let b = v(front_top_right, green, tex.top_right, 0.6);
            let c = v(front_bottom_right, green, tex.bottom_right, 0.6);
            let d = v(front_bottom_left, green, tex.bottom_left, 0.6);
            indices.extend(&[a, b, c, c, d, a]);
        }
        //right face
        if faces_to_render.contains(Faces::Right) {
            let tex = block_texture.right[texture_variation as usize];
            let a = v(front_top_right, red, tex.top_left, 0.6);
            let b = v(back_top_right, red, tex.top_right, 0.6);
            let c = v(back_bottom_right, red, tex.bottom_right, 0.6);
            let d = v(front_bottom_right, red, tex.bottom_left, 0.6);
            indices.extend(&[a, b, c, c, d, a]);
        }
        //left face
        if faces_to_render.contains(Faces::Left) {
            let tex = block_texture.left[texture_variation as usize];
            let a = v(back_top_left, red, tex.top_left, 0.3);
            let b = v(front_top_left, red, tex.top_right, 0.3);
            let c = v(front_bottom_left, red, tex.bottom_right, 0.3);
            let d = v(back_bottom_left, red, tex.bottom_left, 0.3);
            indices.extend(&[a, b, c, c, d, a]);
        }
        //back face
        if faces_to_render.contains(Faces::Back) {
            let tex = block_texture.back[texture_variation as usize];
            let a = v(back_top_right, red, tex.top_left, 0.3);
            let b = v(back_top_left, red, tex.top_right, 0.3);
            let c = v(back_bottom_left, red, tex.bottom_right, 0.3);
            let d = v(back_bottom_right, red, tex.bottom_left, 0.3);
            indices.extend(&[a, b, c, c, d, a]);
        }
        //top face
        if faces_to_render.contains(Faces::Top) {
            let tex = block_texture.top[texture_variation as usize];
            let a = v(back_top_left, red, tex.top_left, 1.0);
            let b = v(back_top_right, red, tex.top_right, 1.0);
            let c = v(front_top_right, red, tex.bottom_right, 1.0);
            let d = v(front_top_left, red, tex.bottom_left, 1.0);
            indices.extend(&[a, b, c, c, d, a]);
        }
        //bottom face
        if faces_to_render.contains(Faces::Bottom) {
            let tex = block_texture.bottom[texture_variation as usize];
            let a = v(front_bottom_left, red, tex.top_left, 0.1);
            let b = v(front_bottom_right, red, tex.top_right, 0.1);
            let c = v(back_bottom_right, red, tex.bottom_right, 0.1);
            let d = v(back_bottom_left, red, tex.bottom_left, 0.1);
            indices.extend(&[a, b, c, c, d, a]);
        }

        let indices: Vec<u32> = indices.iter().map(|x| x + last_index).collect();

        return (vertices, indices);
    }

    fn get_block_texture(block_type: BlockType) -> BlockTypeTexture {
        match block_type {
            BlockType::Air => {
                panic!(
                    "Tried to get a texture for a block type AIR, which should never be rendered!"
                );
            }
            BlockType::Dirt => BLOCK_TEXTURE_DIRT,
            BlockType::Grass => BLOCK_TEXTURE_GRASS,
            BlockType::Sand => BLOCK_TEXTURE_SAND,
            BlockType::Stone => {
                unimplemented!("Stone is not renderable yet, needs art")
            }
        }
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
        if y > CHUNK_MAXIDX_Y as i32 {
            return BlockType::Air;
        }
        //edge case: bottom-later check: Anything under 0 is air.
        if y < 0 {
            return BlockType::Air;
        }

        let mut actual_chunk_z = chunk.chunk_pos.z;
        let mut actual_chunk_x = chunk.chunk_pos.x;

        let mut actual_z = z;
        let mut actual_x = x;

        //Chunk edge detection
        //If X and Z are negative or above their limits, then must check neighboring chunk.
        //First detect for X:
        if x < 0 {
            //x = -1, should be +15 on prev chunk
            actual_chunk_x -= 1; //[-1 0]
            actual_x = CHUNK_SIZE_X as i32 + x;
        } else if x > CHUNK_MAXIDX_X as i32 {
            actual_chunk_x += 1;
            //in this case we need to get the index in the next chunk.
            //If we passed 16 it would be 0 in the next one.
            actual_x = x - CHUNK_SIZE_X as i32
        }

        //Detect for Z
        if z < 0 {
            //z = -1, should be +15 on prev chunk
            actual_chunk_z -= 1;
            actual_z = CHUNK_SIZE_Z as i32 + z
        } else if z > CHUNK_MAXIDX_Z as i32 {
            actual_chunk_z += 1;
            actual_z = z - CHUNK_SIZE_Z as i32;
        }

        //are we sill in the same chunk?

        let same_chunk = actual_chunk_x == chunk.chunk_pos.x && actual_chunk_z == chunk.chunk_pos.z;
        if same_chunk {
            //y z x
            chunk.blocks[y as usize][actual_z as usize][actual_x as usize]
        } else {
            let chunk_find = self.get_chunk(chunkpos(actual_chunk_x, actual_chunk_z));
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

        for (_, chunk) in self.chunks.iter() {
            for (y, layer) in chunk.blocks.iter().enumerate() {
                for (z, row) in layer.iter().enumerate() {
                    for (x, item) in row.iter().enumerate() {
                        if let BlockType::Air = item {
                            continue;
                        }
                        let mut faces_to_render = Faces::None;
                        for (face, offx, offy, offz) in OFFSETS_TO_CHECK {
                            let blocktype = self.blocktype(
                                chunk,
                                x as i32 + offx,
                                y as i32 + offy,
                                z as i32 + offz,
                            );

                            if let BlockType::Air = blocktype {
                                faces_to_render |= face;
                            }
                        }

                        if faces_to_render == Faces::None {
                            continue;
                        }

                        let block_texture = World::get_block_texture(*item);
                        let (generated_vertices, generated_indices) = World::generate_block(
                            chunk.chunk_pos,
                            x,
                            y,
                            z,
                            vertices.len() as u32,
                            block_texture,
                            faces_to_render,
                        );
                        vertices.extend(generated_vertices);
                        indices.extend(generated_indices);
                    }
                }
            }
        }
        return (vertices, indices);
    }

    pub fn get_block_pos(world_pos: Vector3<f32>) -> BlockPositionInWorld {
        return blockpos_world(
            (world_pos.x / BLOCKSIZE) as i32,
            (world_pos.y / BLOCKSIZE) as i32,
            -(world_pos.z / BLOCKSIZE) as i32,
        );
    }

    /*
    block_pos is the position in worldspace, regardless of chunk!
    Should not to be used during world gen
    */
    pub fn set_block(
        &mut self,
        block_pos: BlockPositionInWorld,
        block_type: BlockType,
    ) -> ChunkPosition {
        let chunk_pos = Chunk::block_to_chunk_coords(block_pos);
        let chunk = self.chunks.get_mut(&chunk_pos);
        let chunk_coords =
            Chunk::get_chunkspace_coords_from_worldspace_coords(chunk_pos, block_pos);
        match chunk {
            None => {
                let mut chunk = Chunk::new(chunk_pos);
                chunk.set_block(chunk_coords, block_type);
                self.chunks.insert(chunk.chunk_pos, chunk);
            }
            Some(existing_chunk) => {
                existing_chunk.set_block(chunk_coords, block_type);
            }
        }
        return chunk_pos;
    }

    pub fn set_block_known_chunk(
        &mut self,
        block_pos: BlockPositionInWorld,
        block_type: BlockType,
        chunk: &mut Chunk,
    ) {
        let chunk_pos = Chunk::block_to_chunk_coords(block_pos);
        let chunk_coords =
            Chunk::get_chunkspace_coords_from_worldspace_coords(chunk_pos, block_pos);
        chunk.set_block(chunk_coords, block_type);
    }

    pub fn ensure_chunk(&mut self, chunk_pos: ChunkPosition) {
        if !self.chunks.contains_key(&chunk_pos) {
            self.chunks.insert(chunk_pos, Chunk::new(chunk_pos));
        }
    }

    /*
    pub fn worldgen() -> World {
        use perlin_noise::PerlinNoise;
        let terrain_max_height = PerlinNoise::new();
        terrain_max_height.
        let max_height = CHUNK_SIZE_Y as f64;
        let mut max_height_perlin = 0.012;

        let world_perlin = PerlinNoise::new();
        let mut world = World { chunks: std::collections::HashMap::new() };
        let mut perlin_point = [0.0, 0.0];
        let mut blocks = 0;

        let mut min = i32::MAX;
        let mut max = i32::MIN;

        for z in 0..500 {
            for x in 0..500 {
                blocks += 1;
                let chunk_pos = Chunk::get_chunk_coords(x, z);
                world.ensure_chunk(chunk_pos);
                let chunk = world.expect_chunk_mut(chunk_pos);
                let height_multiply = 1.0 * max_height;

                let perlin = (world_perlin.get2d(perlin_point) * height_multiply).abs();
                let height = perlin as i32;

                if height < min { min = height };
                if height > max { max = height };

                let chunk_coords = Chunk::get_chunkspace_coords_from_worldspace_coords(chunk_pos, blockpos_world(x, height, z));
                chunk.set_block(chunk_coords, BlockType::Grass);
                for y in 0 .. height {
                    let chunk_coords = Chunk::get_chunkspace_coords_from_worldspace_coords(chunk_pos, blockpos_world(x, y, z));
                    chunk.set_block(chunk_coords, BlockType::Dirt);
                    blocks += 1;
                }

                perlin_point[0] += 0.015;

            }
            perlin_point[1] += 0.015;
            perlin_point[0] = 0.0;
            max_height_perlin += 0.001;
        }

        println!("Generated world, chunks: {:?}, blocks: {}", world.chunks.len(), blocks);

        return world;
    }
    */

    pub fn worldgen() -> World {
        let world_perlin = PerlinNoiseGen::new(&TERRAIN);

        let mut world = World {
            chunks: std::collections::HashMap::new(),
        };
        let mut perlin_point = [0.0, 0.0];
        let mut blocks = 0;

        let mut min = i32::MAX;
        let mut max = i32::MIN;

        for z in 0..500 {
            for x in 0..500 {
                blocks += 1;
                let chunk_pos = Chunk::get_chunk_coords(x, z);
                world.ensure_chunk(chunk_pos);
                let chunk = world.expect_chunk_mut(chunk_pos);
                let perlin = world_perlin.sample([x, z]) * (CHUNK_SIZE_Y as f64 / 3.0);
                let height = perlin as i32;

                if height < min {
                    min = height
                };
                if height > max {
                    max = height
                };

                let chunk_coords = Chunk::get_chunkspace_coords_from_worldspace_coords(
                    chunk_pos,
                    blockpos_world(x, height, z),
                );
                chunk.set_block(chunk_coords, BlockType::Grass);
                for y in 0..height {
                    let chunk_coords = Chunk::get_chunkspace_coords_from_worldspace_coords(
                        chunk_pos,
                        blockpos_world(x, y, z),
                    );
                    chunk.set_block(chunk_coords, BlockType::Dirt);
                    blocks += 1;
                }
            }
        }

        println!(
            "Generated world, chunks: {:?}, blocks: {}",
            world.chunks.len(),
            blocks
        );

        return world;
    }
}

//const SAMPLING_RATES: &'static [f64] = &[
//    0.003, 0.01, 0.05
//];

//the highest the value, the more spiky/noisy it is
//the higher the index, the less it contributes to the terrain
const SMOOTH_MOUNTAINS: &'static [f64] = &[0.01];
const INTERESTING_MOUNTAINS: &'static [f64] = &[0.01, 0.04];
const RUGGED_MOUNTAINS: &'static [f64] = &[0.01, 0.04, 0.08];
const NICE_PLAINS: &'static [f64] = &[0.0005];
const MAYBE_NICER_PLAINS: &'static [f64] = &[0.00005, 0.0005, 0.005];
const NICE_PLAINS_2: &'static [f64] = &[0.0005, 0.002];
const LOW_HILLS: &'static [f64] = &[0.0005, 0.01];
const LOW_HILLS_2: &'static [f64] = &[0.0005, 0.02, 0.001];
const HIGH_SPIKES: &'static [f64] = &[0.05, 0.003];
const TERRAIN: &'static [f64] = INTERESTING_MOUNTAINS;

const STARTING_RATE: f64 = 0.5;
const DIVISOR: f64 = 3.0;

struct PerlinNoiseGen {
    sampling_rates: &'static [f64],
    noise_gens: Vec<noise::Perlin>,
}

impl PerlinNoiseGen {
    fn new(sampling_rates: &'static [f64]) -> PerlinNoiseGen {
        let mut noise_gens = vec![noise::Perlin::new(); sampling_rates.len()];
        for (index, g) in noise_gens.iter_mut().enumerate() {
            g.set_seed(index as u32);
        }

        PerlinNoiseGen {
            noise_gens,
            sampling_rates,
        }
    }

    fn sample(&self, block_pos: [i32; 2]) -> f64 {
        let mut rate = STARTING_RATE;
        let mut sampled = 0.0;
        for (gen, sampling_rate) in self.noise_gens.iter().zip(self.sampling_rates) {
            let perlin_coord = [
                block_pos[0] as f64 * sampling_rate,
                block_pos[1] as f64 * sampling_rate,
            ];
            let generated = gen.get(perlin_coord).abs() * rate;
            sampled += generated;
            rate /= DIVISOR;
        }
        return sampled;
    }
}
