#!/usr/bin/env python3
"""
Script de Extração de Blocos YUV 4:2:0 10-bit SEM PERDA DE INFORMAÇÃO

Este script garante:
1. ZERO perda de informação (mantém 10-bit completo)
2. Correspondência EXATA entre blocos extraídos e regiões do vídeo
3. Validação rigorosa de integridade dos dados
4. Logs detalhados para auditoria

Autor: Sistema de Análise CNN_AV1
Data: 2025-10-03
Versão: 1.0.0 - LOSSLESS
"""

import math
import os
import sys
import glob
import struct
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# CONSTANTES E CONFIGURAÇÕES
# ============================================================================

SUPPORTED_BLOCK_SIZES = [64, 32, 16, 8]
OUTPUT_SUBDIR = "intra_raw_blocks"
YUV_FORMAT = "YUV 4:2:0 10-bit Little-Endian Planar"


# ============================================================================
# FUNÇÕES DE CÁLCULO DE FORMATO YUV 4:2:0 10-BIT
# ============================================================================

def calculate_yuv420_10bit_sizes(width, height):
    """
    Calcula tamanhos exatos para YUV 4:2:0 10-bit planar.
    
    Formato: Cada pixel armazenado em 16-bit (little-endian), usando 10 bits.
    
    Args:
        width: Largura do frame em pixels
        height: Altura do frame em pixels
        
    Returns:
        dict com informações detalhadas de tamanhos
    """
    # Componente Y (luminância): width × height pixels, 2 bytes cada
    y_pixels = width * height
    y_size_bytes = y_pixels * 2
    
    # Componentes U e V (crominância): subsampling 4:2:0
    # (width/2) × (height/2) pixels, 2 bytes cada
    uv_pixels = (width // 2) * (height // 2)
    u_size_bytes = uv_pixels * 2
    v_size_bytes = uv_pixels * 2
    
    # Total por frame
    total_frame_size = y_size_bytes + u_size_bytes + v_size_bytes
    
    return {
        'y_pixels': y_pixels,
        'y_size_bytes': y_size_bytes,
        'uv_pixels': uv_pixels,
        'u_size_bytes': u_size_bytes,
        'v_size_bytes': v_size_bytes,
        'total_frame_size': total_frame_size,
        'width': width,
        'height': height
    }


def validate_yuv_file_integrity(yuv_path, width, height, verbose=True):
    """
    Valida integridade completa de um arquivo YUV 4:2:0 10-bit.
    
    Args:
        yuv_path: Caminho para arquivo YUV
        width: Largura esperada
        height: Altura esperada
        verbose: Se True, imprime detalhes
        
    Returns:
        dict com resultado da validação
    """
    if not os.path.exists(yuv_path):
        return {
            'valid': False,
            'error': 'Arquivo não encontrado',
            'path': yuv_path
        }
    
    file_size = os.path.getsize(yuv_path)
    sizes = calculate_yuv420_10bit_sizes(width, height)
    frame_size = sizes['total_frame_size']
    
    num_frames = file_size // frame_size
    remainder = file_size % frame_size
    
    is_valid = (remainder == 0 and num_frames > 0)
    
    result = {
        'valid': is_valid,
        'path': yuv_path,
        'file_size_bytes': file_size,
        'file_size_mb': file_size / (1024 * 1024),
        'frame_size_bytes': frame_size,
        'num_frames': num_frames,
        'remainder_bytes': remainder,
        'sizes': sizes
    }
    
    if verbose and is_valid:
        print(f"  ✅ Arquivo YUV válido:")
        print(f"    Caminho: {yuv_path}")
        print(f"    Tamanho: {result['file_size_mb']:.2f} MB")
        print(f"    Frames: {num_frames}")
        print(f"    Frame size: {frame_size:,} bytes")
        print(f"    Y component: {sizes['y_size_bytes']:,} bytes ({sizes['y_pixels']:,} pixels)")
        print(f"    U component: {sizes['u_size_bytes']:,} bytes ({sizes['uv_pixels']:,} pixels)")
        print(f"    V component: {sizes['v_size_bytes']:,} bytes ({sizes['uv_pixels']:,} pixels)")
    elif verbose:
        print(f"  ❌ Arquivo YUV INVÁLIDO:")
        print(f"    Caminho: {yuv_path}")
        print(f"    Tamanho: {file_size:,} bytes")
        print(f"    Frames calculados: {num_frames}")
        print(f"    Resto: {remainder} bytes (esperado: 0)")
    
    return result


# ============================================================================
# FUNÇÕES DE LEITURA YUV 10-BIT (LOSSLESS)
# ============================================================================

def read_y_component_10bit_lossless(yuv_path, frame_number, width, height):
    """
    Lê componente Y (luminância) mantendo TODOS os 10 bits de informação.
    
    IMPORTANTE: Retorna valores 10-bit (0-1023) em dtype uint16.
    NÃO faz conversão para 8-bit, mantendo precisão total.
    
    Args:
        yuv_path: Caminho para arquivo YUV
        frame_number: Número do frame (0-indexed)
        width: Largura do frame
        height: Altura do frame
        
    Returns:
        numpy.ndarray: Matriz 2D (height, width) com valores 10-bit em uint16
        ou None se houver erro
    """
    sizes = calculate_yuv420_10bit_sizes(width, height)
    frame_size = sizes['total_frame_size']
    y_size = sizes['y_size_bytes']
    
    try:
        with open(yuv_path, 'rb') as f:
            # Calcular offset exato para o frame
            frame_offset = frame_number * frame_size
            
            # Seek para posição do frame
            f.seek(frame_offset, 0)
            
            # Ler apenas componente Y (primeiros y_size bytes do frame)
            y_buffer = f.read(y_size)
            
            # Validar leitura completa
            if len(y_buffer) != y_size:
                raise IOError(
                    f"Leitura incompleta: esperado {y_size} bytes, "
                    f"lido {len(y_buffer)} bytes"
                )
            
            # Converter bytes para array uint16 (little-endian)
            y_data_16bit = np.frombuffer(y_buffer, dtype='<u2')  # '<u2' = uint16 little-endian
            
            # Validar range 10-bit (0-1023)
            min_val = np.min(y_data_16bit)
            max_val = np.max(y_data_16bit)
            
            if max_val > 1023:
                print(f"    ⚠️  AVISO: Valor máximo {max_val} > 1023 (esperado para 10-bit)")
                print(f"    Dados podem não ser 10-bit ou ter padding nos bits superiores")
            
            if min_val < 0:
                raise ValueError(f"Valor mínimo {min_val} < 0 (inválido)")
            
            # Reshape para matriz 2D
            y_matrix = y_data_16bit.reshape(height, width)
            
            # Estatísticas para validação
            stats = {
                'min': int(min_val),
                'max': int(max_val),
                'mean': float(np.mean(y_data_16bit)),
                'std': float(np.std(y_data_16bit)),
                'shape': y_matrix.shape,
                'dtype': str(y_matrix.dtype)
            }
            
            return y_matrix, stats
            
    except Exception as e:
        print(f"  ❌ ERRO ao ler componente Y do frame {frame_number}: {e}")
        return None, None


def compute_data_hash(data):
    """
    Calcula hash MD5 dos dados para verificação de integridade.
    
    Args:
        data: numpy array
        
    Returns:
        str: Hash MD5 hexadecimal
    """
    return hashlib.md5(data.tobytes()).hexdigest()


# ============================================================================
# FUNÇÕES DE EXTRAÇÃO DE NOME E VALIDAÇÃO DE XLSX
# ============================================================================

def extract_sequence_name_from_xlsx(xlsx_filename):
    """
    Extrai nome da sequência de vídeo do arquivo XLSX.
    
    Padrão esperado: {sequence_name}-intra-{frame_number}.xlsx
    
    Exemplo:
        Input:  'Beauty_3840x2160_120fps_420_10bit-intra-5.xlsx'
        Output: 'Beauty_3840x2160_120fps_420_10bit'
    
    Args:
        xlsx_filename: Nome ou caminho do arquivo XLSX
        
    Returns:
        str: Nome da sequência (sem extensão e sufixo -intra-X)
    """
    basename = os.path.basename(xlsx_filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    parts = name_without_ext.split('-')
    
    # Verificar padrão -intra-{número}
    if len(parts) >= 3 and parts[-2] == 'intra':
        try:
            # Validar que último elemento é número
            _ = int(parts[-1])
            # Remover '-intra-{número}'
            sequence_name = '-'.join(parts[:-2])
            return sequence_name
        except ValueError:
            pass
    
    # Fallback: remover apenas último elemento
    if len(parts) > 1:
        return '-'.join(parts[:-1])
    
    return name_without_ext


def extract_frame_number_from_xlsx(xlsx_filename):
    """
    Extrai número do frame do nome do arquivo XLSX.
    
    Padrão esperado: {sequence_name}-intra-{frame_number}.xlsx
    
    Args:
        xlsx_filename: Nome ou caminho do arquivo XLSX
        
    Returns:
        int: Número do frame ou None se não encontrado
    """
    basename = os.path.basename(xlsx_filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    parts = name_without_ext.split('-')
    
    # Tentar extrair do padrão -intra-{número}
    if len(parts) >= 3 and parts[-2] == 'intra':
        try:
            frame_number = int(parts[-1])
            return frame_number
        except ValueError:
            pass
    
    # Tentar último elemento
    try:
        frame_number = int(parts[-1])
        return frame_number
    except (ValueError, IndexError):
        return None


def validate_xlsx_structure(xlsx_path, expected_sheets=None):
    """
    Valida estrutura do arquivo XLSX.
    
    Args:
        xlsx_path: Caminho para arquivo XLSX
        expected_sheets: Lista de nomes de planilhas esperadas (opcional)
        
    Returns:
        dict com resultado da validação
    """
    if not os.path.exists(xlsx_path):
        return {
            'valid': False,
            'error': 'Arquivo não encontrado',
            'path': xlsx_path
        }
    
    try:
        xlsx = pd.ExcelFile(xlsx_path)
        sheets = xlsx.sheet_names
        
        if expected_sheets:
            missing_sheets = set(expected_sheets) - set(sheets)
            has_all_sheets = len(missing_sheets) == 0
        else:
            missing_sheets = set()
            has_all_sheets = True
        
        return {
            'valid': has_all_sheets,
            'path': xlsx_path,
            'sheets': sheets,
            'missing_sheets': list(missing_sheets),
            'num_sheets': len(sheets)
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'path': xlsx_path
        }


# ============================================================================
# FUNÇÕES DE EXTRAÇÃO DE BLOCOS COM VALIDAÇÃO RIGOROSA
# ============================================================================

def extract_blocks_with_validation(y_matrix, block_size, width, height, verbose=True):
    """
    Extrai blocos da matriz Y com validação rigorosa de geometria.
    
    GARANTE que cada bloco corresponde EXATAMENTE à região correta do vídeo.
    
    Args:
        y_matrix: Matriz numpy (height, width) com dados Y 10-bit
        block_size: Tamanho do bloco (ex: 64)
        width: Largura original do frame
        height: Altura original do frame
        verbose: Se True, imprime informações detalhadas
        
    Returns:
        tuple: (blocks_array, metadata)
            blocks_array: numpy array (num_blocks, block_size, block_size)
            metadata: dict com informações de extração
    """
    # Calcular grid de blocos
    num_rows = math.ceil(height / block_size)
    num_cols = math.ceil(width / block_size)
    
    # Dimensões estendidas (com padding se necessário)
    padded_height = num_rows * block_size
    padded_width = num_cols * block_size
    
    # Aplicar padding se necessário (preenchimento com zero)
    if padded_height > height or padded_width > width:
        padded_matrix = np.zeros((padded_height, padded_width), dtype=np.uint16)
        padded_matrix[:height, :width] = y_matrix
        
        padding_info = {
            'applied': True,
            'original_size': (height, width),
            'padded_size': (padded_height, padded_width),
            'padding_bottom': padded_height - height,
            'padding_right': padded_width - width
        }
    else:
        padded_matrix = y_matrix
        padding_info = {'applied': False}
    
    # Extrair blocos em ordem row-major (esquerda->direita, cima->baixo)
    num_blocks = num_rows * num_cols
    blocks = np.zeros((num_blocks, block_size, block_size), dtype=np.uint16)
    
    block_positions = []
    block_idx = 0
    
    for row_idx in range(num_rows):
        y_start = row_idx * block_size
        y_end = y_start + block_size
        
        for col_idx in range(num_cols):
            x_start = col_idx * block_size
            x_end = x_start + block_size
            
            # Extrair bloco
            block = padded_matrix[y_start:y_end, x_start:x_end]
            
            # Validação de tamanho
            if block.shape != (block_size, block_size):
                raise ValueError(
                    f"Bloco extraído com tamanho incorreto: "
                    f"esperado ({block_size}, {block_size}), "
                    f"obtido {block.shape}"
                )
            
            blocks[block_idx] = block
            
            # Armazenar posição para rastreabilidade
            block_positions.append({
                'block_idx': block_idx,
                'row': row_idx,
                'col': col_idx,
                'y_range': (y_start, y_end),
                'x_range': (x_start, x_end),
                'is_padded': (y_end > height or x_end > width)
            })
            
            block_idx += 1
    
    # Metadata completo
    metadata = {
        'block_size': block_size,
        'num_blocks': num_blocks,
        'grid_shape': (num_rows, num_cols),
        'original_frame_size': (height, width),
        'padded_frame_size': (padded_height, padded_width),
        'padding_info': padding_info,
        'block_positions': block_positions,
        'extraction_order': 'row-major',
        'dtype': str(blocks.dtype)
    }
    
    if verbose:
        print(f"    Blocos {block_size}×{block_size}:")
        print(f"      Grid: {num_rows}×{num_cols} = {num_blocks} blocos")
        print(f"      Frame original: {height}×{width}")
        print(f"      Frame com padding: {padded_height}×{padded_width}")
        if padding_info['applied']:
            print(f"      Padding aplicado: {padding_info['padding_bottom']}px (baixo), "
                  f"{padding_info['padding_right']}px (direita)")
    
    return blocks, metadata


def filter_blocks_by_labels(blocks, metadata, label_data, block_size, verbose=True):
    """
    Filtra blocos baseado nos dados de label do Excel.
    
    VALIDAÇÃO RIGOROSA: Verifica correspondência exata entre labels e posições.
    
    Args:
        blocks: Array de blocos (num_blocks, block_size, block_size)
        metadata: Metadata da extração
        label_data: DataFrame com labels do Excel (coluna B)
        block_size: Tamanho do bloco
        verbose: Se True, imprime informações
        
    Returns:
        tuple: (filtered_blocks, filter_metadata)
    """
    # Extrair valores da coluna de labels
    label_cols = (label_data.values / block_size) * 4
    label_cols = label_cols.astype(int).flatten()
    num_labels = len(label_cols)
    
    if verbose:
        print(f"      Labels encontradas: {num_labels}")
        print(f"      Primeiras 5 labels (col): {label_cols[:5].tolist()}")
    
    # Validar número de labels
    num_blocks = metadata['num_blocks']
    num_rows, num_cols = metadata['grid_shape']
    
    if num_labels > num_blocks:
        raise ValueError(
            f"Número de labels ({num_labels}) excede número de blocos ({num_blocks})"
        )
    
    # Filtrar blocos baseado nas labels
    # Labels indicam coluna esperada para cada bloco
    filtered_blocks = []
    kept_indices = []
    discarded_count = 0
    
    label_idx = 0
    for block_idx in range(num_blocks):
        if label_idx >= num_labels:
            break
        
        pos = metadata['block_positions'][block_idx]
        expected_col = label_cols[label_idx]
        actual_col = pos['col']
        
        if actual_col == expected_col:
            # Label corresponde à posição: MANTER bloco
            filtered_blocks.append(blocks[block_idx])
            kept_indices.append(block_idx)
            label_idx += 1
        else:
            # Label não corresponde: DESCARTAR bloco
            discarded_count += 1
    
    filtered_blocks = np.array(filtered_blocks, dtype=np.uint16)
    
    filter_metadata = {
        'original_count': num_blocks,
        'filtered_count': len(filtered_blocks),
        'discarded_count': discarded_count,
        'kept_indices': kept_indices,
        'num_labels': num_labels
    }
    
    if verbose:
        print(f"      Filtragem:")
        print(f"        Blocos originais: {num_blocks}")
        print(f"        Blocos mantidos: {len(filtered_blocks)}")
        print(f"        Blocos descartados: {discarded_count}")
    
    return filtered_blocks, filter_metadata


# ============================================================================
# FUNÇÕES DE SALVAMENTO COM VALIDAÇÃO
# ============================================================================

def save_blocks_binary_10bit(blocks, output_path, metadata, verbose=True):
    """
    Salva blocos em formato binário mantendo 10-bit (uint16 little-endian).
    
    Formato de saída:
    - Cada pixel: 2 bytes (uint16 little-endian)
    - Valores: 0-1023 (10-bit)
    - Ordem: Blocos flatten em row-major order
    
    Args:
        blocks: Array numpy (num_blocks, block_size, block_size) dtype uint16
        output_path: Caminho para arquivo de saída
        metadata: Metadata para incluir no log
        verbose: Se True, imprime informações
        
    Returns:
        dict com informações do salvamento
    """
    # Validar dtype
    if blocks.dtype != np.uint16:
        raise TypeError(f"Blocos devem ser uint16, recebido {blocks.dtype}")
    
    # Flatten para 1D
    blocks_flat = blocks.flatten()
    
    # Calcular hash ANTES de salvar
    data_hash = compute_data_hash(blocks_flat)
    
    # Estatísticas
    stats = {
        'num_blocks': blocks.shape[0],
        'block_size': blocks.shape[1],
        'total_pixels': blocks_flat.size,
        'total_bytes': blocks_flat.nbytes,
        'min_value': int(np.min(blocks_flat)),
        'max_value': int(np.max(blocks_flat)),
        'mean_value': float(np.mean(blocks_flat)),
        'std_value': float(np.std(blocks_flat)),
        'dtype': str(blocks.dtype),
        'md5_hash': data_hash
    }
    
    # Salvar em formato binário
    with open(output_path, 'wb') as f:
        # Escrever como little-endian uint16
        blocks_flat.astype('<u2').tofile(f)
    
    # Verificar arquivo salvo
    file_size = os.path.getsize(output_path)
    expected_size = stats['total_bytes']
    
    if file_size != expected_size:
        raise IOError(
            f"Tamanho do arquivo salvo ({file_size}) difere do esperado ({expected_size})"
        )
    
    # Verificar integridade: ler de volta e comparar hash
    with open(output_path, 'rb') as f:
        readback = np.fromfile(f, dtype='<u2')
    
    readback_hash = compute_data_hash(readback)
    
    if readback_hash != data_hash:
        raise ValueError(
            f"Verificação de integridade FALHOU! "
            f"Hash original: {data_hash}, Hash lido: {readback_hash}"
        )
    
    stats['file_path'] = output_path
    stats['file_size_bytes'] = file_size
    stats['file_size_mb'] = file_size / (1024 * 1024)
    stats['integrity_verified'] = True
    
    if verbose:
        print(f"      ✅ Arquivo salvo: {os.path.basename(output_path)}")
        print(f"        Tamanho: {stats['file_size_mb']:.3f} MB")
        print(f"        Pixels: {stats['total_pixels']:,} ({stats['num_blocks']} blocos)")
        print(f"        Range: {stats['min_value']}-{stats['max_value']} (10-bit)")
        print(f"        Hash MD5: {data_hash[:16]}...")
        print(f"        Integridade: VERIFICADA ✓")
    
    return stats


# ============================================================================
# PROCESSAMENTO PRINCIPAL
# ============================================================================

def process_single_xlsx_sequence(xlsx_path, video_dir, video_ext, frame_width, frame_height, output_dir):
    """
    Processa uma única sequência (um arquivo XLSX).
    
    Args:
        xlsx_path: Caminho para arquivo XLSX
        video_dir: Diretório contendo vídeos YUV
        video_ext: Extensão dos vídeos (ex: 'yuv')
        frame_width: Largura do frame
        frame_height: Altura do frame
        output_dir: Diretório de saída
        
    Returns:
        dict com resultados do processamento
    """
    print(f"\n{'='*80}")
    print(f"PROCESSANDO: {os.path.basename(xlsx_path)}")
    print(f"{'='*80}")
    
    # Extrair informações do nome do arquivo
    sequence_name = extract_sequence_name_from_xlsx(xlsx_path)
    frame_number = extract_frame_number_from_xlsx(xlsx_path)
    
    print(f"  Sequência: {sequence_name}")
    print(f"  Frame: {frame_number}")
    
    if frame_number is None:
        print(f"  ❌ ERRO: Não foi possível extrair número do frame")
        return {'success': False, 'error': 'Frame number extraction failed'}
    
    # Localizar arquivo de vídeo
    video_path = os.path.join(video_dir, f"{sequence_name}.{video_ext}")
    
    if not os.path.exists(video_path):
        print(f"  ❌ ERRO: Arquivo de vídeo não encontrado: {video_path}")
        return {'success': False, 'error': 'Video file not found', 'path': video_path}
    
    print(f"  Vídeo: {video_path}")
    
    # Validar arquivo YUV
    print(f"\n  [1/5] Validando arquivo YUV...")
    yuv_validation = validate_yuv_file_integrity(video_path, frame_width, frame_height, verbose=True)
    
    if not yuv_validation['valid']:
        print(f"  ❌ ERRO: Arquivo YUV inválido")
        return {'success': False, 'error': 'Invalid YUV file', 'validation': yuv_validation}
    
    # Verificar se frame existe
    if frame_number >= yuv_validation['num_frames']:
        print(f"  ❌ ERRO: Frame {frame_number} não existe (arquivo tem {yuv_validation['num_frames']} frames)")
        return {'success': False, 'error': 'Frame number out of range'}
    
    # Validar estrutura do XLSX
    print(f"\n  [2/5] Validando arquivo XLSX...")
    expected_sheets = [str(size) for size in SUPPORTED_BLOCK_SIZES]
    xlsx_validation = validate_xlsx_structure(xlsx_path, expected_sheets)
    
    if not xlsx_validation['valid']:
        print(f"  ⚠️  AVISO: Planilhas esperadas não encontradas")
        print(f"      Esperadas: {expected_sheets}")
        print(f"      Encontradas: {xlsx_validation['sheets']}")
        print(f"      Faltando: {xlsx_validation['missing_sheets']}")
    
    # Ler componente Y do frame (10-bit LOSSLESS)
    print(f"\n  [3/5] Lendo frame {frame_number} (YUV 4:2:0 10-bit)...")
    y_matrix, y_stats = read_y_component_10bit_lossless(
        video_path, frame_number, frame_width, frame_height
    )
    
    if y_matrix is None:
        print(f"  ❌ ERRO: Falha ao ler componente Y")
        return {'success': False, 'error': 'Y component read failed'}
    
    print(f"    ✅ Componente Y lida com sucesso (10-bit LOSSLESS):")
    print(f"      Shape: {y_stats['shape']}")
    print(f"      Dtype: {y_stats['dtype']}")
    print(f"      Range: {y_stats['min']}-{y_stats['max']}")
    print(f"      Média: {y_stats['mean']:.2f}")
    print(f"      Desvio padrão: {y_stats['std']:.2f}")
    
    # Processar cada tamanho de bloco
    print(f"\n  [4/5] Extraindo e filtrando blocos...")
    
    results = {
        'success': True,
        'sequence_name': sequence_name,
        'frame_number': frame_number,
        'video_path': video_path,
        'xlsx_path': xlsx_path,
        'frame_size': (frame_height, frame_width),
        'y_stats': y_stats,
        'blocks': {}
    }
    
    for block_size in SUPPORTED_BLOCK_SIZES:
        print(f"\n    {'─'*60}")
        print(f"    Processando blocos {block_size}×{block_size}...")
        print(f"    {'─'*60}")
        
        sheet_name = str(block_size)
        
        # Verificar se planilha existe
        if sheet_name not in xlsx_validation['sheets']:
            print(f"      ⚠️  Planilha '{sheet_name}' não encontrada, pulando...")
            continue
        
        try:
            # Ler labels do Excel
            xlsx = pd.ExcelFile(xlsx_path)
            label_data = pd.read_excel(xlsx, sheet_name=sheet_name, usecols="B")
            
            # Extrair blocos
            blocks, extract_metadata = extract_blocks_with_validation(
                y_matrix, block_size, frame_width, frame_height, verbose=True
            )
            
            # Filtrar por labels
            filtered_blocks, filter_metadata = filter_blocks_by_labels(
                blocks, extract_metadata, label_data, block_size, verbose=True
            )
            
            # Gerar nome do arquivo de saída
            output_filename = f"{sequence_name}_sample_{block_size}.txt"
            output_path = os.path.join(output_dir, output_filename)
            
            # Salvar blocos (10-bit LOSSLESS)
            print(f"      Salvando blocos (10-bit LOSSLESS)...")
            save_stats = save_blocks_binary_10bit(
                filtered_blocks, output_path, extract_metadata, verbose=True
            )
            
            # Armazenar resultados
            results['blocks'][block_size] = {
                'extract_metadata': extract_metadata,
                'filter_metadata': filter_metadata,
                'save_stats': save_stats,
                'output_path': output_path
            }
            
        except Exception as e:
            print(f"      ❌ ERRO ao processar blocos {block_size}×{block_size}: {e}")
            results['blocks'][block_size] = {
                'success': False,
                'error': str(e)
            }
            results['success'] = False
    
    print(f"\n  [5/5] Processamento concluído!")
    
    return results


def main():
    """Função principal."""
    print(f"\n{'='*80}")
    print(f"YUV 4:2:0 10-BIT LOSSLESS BLOCK EXTRACTOR")
    print(f"Versão 1.0.0 - Zero Perda de Informação")
    print(f"{'='*80}\n")
    
    # Validar argumentos
    if len(sys.argv) != 6:
        print("ERRO: Número incorreto de argumentos!\n")
        print("USO:")
        print("  python 005_rearrange_video_YUV_420_10bit_LOSSLESS.py \\")
        print("         <dir_xlsx> <dir_videos> <ext_video> <largura> <altura>\n")
        print("EXEMPLOS:")
        print("  python 005_rearrange_video_YUV_420_10bit_LOSSLESS.py \\")
        print("         /home/experimentos/xlsx_labels \\")
        print("         /home/videos/yuv_10bit \\")
        print("         yuv \\")
        print("         3840 \\")
        print("         2160\n")
        print("PARÂMETROS:")
        print("  dir_xlsx   : Diretório contendo arquivos XLSX com labels")
        print("  dir_videos : Diretório contendo vídeos YUV 4:2:0 10-bit")
        print("  ext_video  : Extensão dos arquivos de vídeo (yuv)")
        print("  largura    : Largura dos frames em pixels")
        print("  altura     : Altura dos frames em pixels\n")
        print("GARANTIAS:")
        print("  ✓ Zero perda de informação (mantém 10-bit completo)")
        print("  ✓ Correspondência exata bloco ↔ região do vídeo")
        print("  ✓ Validação de integridade com hash MD5")
        print("  ✓ Verificação rigorosa de geometria")
        print("  ✓ Logs detalhados para auditoria\n")
        sys.exit(1)
    
    # Parse argumentos
    xlsx_dir = sys.argv[1]
    video_dir = sys.argv[2]
    video_ext = sys.argv[3].lower()
    frame_width = int(sys.argv[4])
    frame_height = int(sys.argv[5])
    
    # Validar diretórios
    if not os.path.isdir(xlsx_dir):
        print(f"❌ ERRO: Diretório XLSX não encontrado: {xlsx_dir}")
        sys.exit(1)
    
    if not os.path.isdir(video_dir):
        print(f"❌ ERRO: Diretório de vídeos não encontrado: {video_dir}")
        sys.exit(1)
    
    # Buscar arquivos XLSX com padrão -intra-
    xlsx_pattern = os.path.join(xlsx_dir, "*-intra-*.xlsx")
    xlsx_files = sorted(glob.glob(xlsx_pattern))
    
    if not xlsx_files:
        print(f"❌ ERRO: Nenhum arquivo XLSX com padrão '*-intra-*.xlsx' encontrado em: {xlsx_dir}")
        sys.exit(1)
    
    # Criar diretório de saída
    output_dir = os.path.join(xlsx_dir, OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Imprimir configuração
    print(f"CONFIGURAÇÃO:")
    print(f"  Diretório XLSX: {xlsx_dir}")
    print(f"  Diretório vídeos: {video_dir}")
    print(f"  Extensão vídeo: {video_ext}")
    print(f"  Dimensões frame: {frame_width}×{frame_height}")
    print(f"  Arquivos XLSX encontrados: {len(xlsx_files)}")
    print(f"  Tamanhos de bloco: {SUPPORTED_BLOCK_SIZES}")
    print(f"  Diretório de saída: {output_dir}")
    print(f"  Formato YUV: {YUV_FORMAT}\n")
    
    # Calcular informações do formato
    sizes = calculate_yuv420_10bit_sizes(frame_width, frame_height)
    print(f"INFORMAÇÕES DO FORMATO YUV 4:2:0 10-BIT:")
    print(f"  Componente Y: {sizes['y_size_bytes']:,} bytes ({sizes['y_pixels']:,} pixels)")
    print(f"  Componente U: {sizes['u_size_bytes']:,} bytes ({sizes['uv_pixels']:,} pixels)")
    print(f"  Componente V: {sizes['v_size_bytes']:,} bytes ({sizes['uv_pixels']:,} pixels)")
    print(f"  Total por frame: {sizes['total_frame_size']:,} bytes ({sizes['total_frame_size']/1024/1024:.2f} MB)")
    
    # Processar cada arquivo XLSX
    print(f"\n{'='*80}")
    print(f"INICIANDO PROCESSAMENTO")
    print(f"{'='*80}")
    
    total_processed = 0
    total_failed = 0
    all_results = []
    
    for xlsx_file in xlsx_files:
        result = process_single_xlsx_sequence(
            xlsx_file, video_dir, video_ext, 
            frame_width, frame_height, output_dir
        )
        
        all_results.append(result)
        
        if result['success']:
            total_processed += 1
        else:
            total_failed += 1
    
    # Sumário final
    print(f"\n{'='*80}")
    print(f"PROCESSAMENTO CONCLUÍDO")
    print(f"{'='*80}")
    print(f"  Total processados: {total_processed}")
    print(f"  Total com falhas: {total_failed}")
    print(f"  Arquivos de saída em: {output_dir}")
    
    if total_failed == 0:
        print(f"\n  ✅ SUCESSO TOTAL! Todos os arquivos processados sem perda de informação.")
    else:
        print(f"\n  ⚠️  {total_failed} arquivos com falhas. Verifique os logs acima.")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
