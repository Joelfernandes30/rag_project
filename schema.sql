-- Enable vector extension
create extension if not exists vector;

-- Recreate vectors table with native Google 768 dimensions
drop table if exists public.charaka_samhita_vectors cascade;

create table public.charaka_samhita_vectors (
  id text primary key,
  content text not null,
  section text,
  chapter text,
  pages text,
  page_number text,
  source_file text,
  chapter_label text,
  embedding vector(768)
);

-- Enable RLS (Row Level Security) and allow read/write
alter table public.charaka_samhita_vectors enable row level security;

create policy "Allow read access to all users"
on public.charaka_samhita_vectors for select
using (true);

create policy "Allow write access to all users"
on public.charaka_samhita_vectors for all
using (true);

-- Create HNSW Index for cosine vector distance
create index charaka_samhita_vectors_embedding_idx 
on public.charaka_samhita_vectors 
using hnsw (embedding vector_cosine_ops);

-- Similarity search function match_documents
create or replace function public.match_documents (
  query_embedding vector(768),
  match_count int default 5
)
returns table (
  id text,
  content text,
  section text,
  chapter text,
  pages text,
  page_number text,
  source_file text,
  chapter_label text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    v.id,
    v.content,
    v.section,
    v.chapter,
    v.pages,
    v.page_number,
    v.source_file,
    v.chapter_label,
    1 - (v.embedding <=> query_embedding) as similarity
  from public.charaka_samhita_vectors v
  order by v.embedding <=> query_embedding
  limit match_count;
end;
$$;
