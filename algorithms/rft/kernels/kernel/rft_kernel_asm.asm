; SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
; Copyright (C) 2025 Luis M. Minier / quantoniumos
; Listed in CLAIMS_PRACTICING_FILES.txt — LICENSE-CLAIMS-NC.md applies.

; rft_kernel_asm.asm
; Assembly optimized routines for CANONICAL RFT kernel
;
; CANONICAL RFT DEFINITION (January 2026):
;   Φ̃ = Φ (ΦᴴΦ)^{-1/2} where Φ[n,k] = exp(j 2π frac((k+1)φ) n) / √N
;
; This is the Gram-normalized φ-grid exponential basis that provides:
;   - Exact unitarity: Φ̃ᴴ Φ̃ = I
;   - +15-20 dB PSNR on golden quasi-periodic signals
;
; See: THEOREMS_RFT_IRONCLAD.md, algorithms/rft/README_RFT.md

section .text
global rft_transform_asm
global rft_basis_multiply_asm
global rft_quantum_gate_asm

; Constants
align 16
rft_consts:
    dq 6.283185307179586    ; 2*PI
    dq 1.618033988749895    ; PHI (golden ratio)
    dq 0.618033988749895    ; 1/PHI = PHI - 1
    dq 1.272019649514069    ; sqrt(phi) - quantum normalization factor

;------------------------------------------------------------------------------
; rft_transform_asm - Optimized unitary RFT transform implementation
;
; Parameters:
;   rdi - input buffer (complex doubles)
;   rsi - output buffer (complex doubles)
;   rdx - basis matrix (complex doubles)
;   rcx - size (number of elements)
;
; Note: Uses the System V AMD64 ABI calling convention
;------------------------------------------------------------------------------
rft_transform_asm:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    ; Save parameters
    mov r8, rdi      ; r8 = input
    mov r9, rsi      ; r9 = output
    mov r10, rdx     ; r10 = basis
    mov r11, rcx     ; r11 = size
    
    ; Zero output buffer first
    xor rax, rax
    mov rcx, r11
    shl rcx, 4       ; rcx = size * 16 (each complex is 16 bytes)
    mov rdi, r9
    rep stosb
    
    ; Main computation loop - true unitary transform
    ; for (i = 0; i < size; i++)
    xor r12, r12     ; r12 = i = 0
.loop_i:
    cmp r12, r11
    jge .done
    
    ; for (j = 0; j < size; j++)
    xor r13, r13     ; r13 = j = 0
.loop_j:
    cmp r13, r11
    jge .next_i
    
    ; Load input[j]
    mov rax, r13
    shl rax, 4       ; rax = j * 16 (each complex is 16 bytes)
    movapd xmm0, [r8 + rax]   ; xmm0 = input[j] (complex)
    
    ; Calculate basis index - for unitary transform
    ; For forward transform with unitary matrix, we use conjugate transpose
    ; So we swap i and j and conjugate the basis element
    mov rax, r13     ; column (j)
    mul r11          ; rax = column * size
    add rax, r12     ; rax = column * size + row = j * size + i
    shl rax, 4       ; rax = (j * size + i) * 16
    
    ; Load basis[j,i] = basis†[i,j]
    movapd xmm1, [r10 + rax]   ; xmm1 = basis[j,i] (complex)
    
    ; Conjugate basis element (negate imaginary part)
    movapd xmm5, xmm1
    movsd xmm6, [rel .conj_mask]
    xorpd xmm5, xmm6    ; Flip sign of imaginary part
    
    ; Complex multiplication:
    ; (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    movapd xmm2, xmm0        ; xmm2 = input[j]
    movapd xmm3, xmm5        ; xmm3 = conjugate of basis[j,i]
    
    ; Shuffle for multiplication
    shufpd xmm2, xmm2, 1     ; xmm2 = [im, re]
    
    ; Multiply
    mulpd xmm0, xmm3         ; xmm0 = [re*re, im*im]
    mulpd xmm2, xmm3         ; xmm2 = [im*re, re*im]
    
    ; Shuffle and subtract for real part
    movapd xmm4, xmm0        ; xmm4 = [re*re, im*im]
    shufpd xmm4, xmm4, 1     ; xmm4 = [im*im, re*re]
    subsd xmm0, xmm4         ; xmm0 = [re*re-im*im (real part), im*im]
    
    ; Add for imaginary part
    addsd xmm2, xmm3         ; xmm2 = [im*re+re*im (imag part), ?]
    
    ; Combine real and imaginary parts
    shufpd xmm0, xmm2, 0     ; xmm0 = [real part, imag part]
    
    ; Add to output[i]
    mov rax, r12
    shl rax, 4               ; rax = i * 16
    movapd xmm4, [r9 + rax]  ; xmm4 = current output[i]
    addpd xmm4, xmm0         ; xmm4 += complex multiplication result
    movapd [r9 + rax], xmm4  ; Store back to output[i]
    
    ; Next j
    inc r13
    jmp .loop_j
    
.next_i:
    inc r12
    jmp .loop_i
    
.done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; Bitmask for conjugation (0x0000000000000000, 0x8000000000000000)
align 16
.conj_mask: dq 0x0000000000000000, 0x8000000000000000

;------------------------------------------------------------------------------
; rft_basis_multiply_asm - Optimized matrix-vector multiplication for RFT
;
; Parameters:
;   rdi - input vector (complex doubles)
;   rsi - output vector (complex doubles)
;   rdx - basis matrix (complex doubles)
;   rcx - size (number of elements)
;   r8  - transpose flag (0 = normal, 1 = conjugate transpose)
;
; Note: Uses the System V AMD64 ABI calling convention
;------------------------------------------------------------------------------
rft_basis_multiply_asm:
    push rbp
    mov rbp, rsp
    
    ; Main computation loop handled by rft_transform_asm
    ; This is just a wrapper with extra transpose flag
    
    ; If transpose flag is set, we need to handle conjugation
    test r8, r8
    jz .normal_mult

    ; Implement conjugate transpose multiplication
    ; For each complex number (real, imag), conjugate means (real, -imag)
    ; We need to negate the imaginary parts before calling transform

    ; Save registers
    push rbx
    push rcx
    push rdx

    ; Get vector length (assume rsi contains length)
    mov rcx, rsi
    shr rcx, 1  ; Divide by 2 for complex pairs

    ; Conjugate the input vector (negate imaginary parts)
    mov rbx, rdi  ; Input vector pointer
.conjugate_loop:
    ; Load complex number (real, imag)
    movsd xmm0, [rbx]      ; real
    movsd xmm1, [rbx + 8]  ; imag

    ; Negate imaginary part for conjugation
    xorpd xmm1, xmm2       ; xmm2 should be set to sign bit mask
    movsd [rbx + 8], xmm1

    ; Move to next complex number
    add rbx, 16
    loop .conjugate_loop

    ; Restore registers
    pop rdx
    pop rcx
    pop rbx

    ; Now call normal transform
    call rft_transform_asm

    ; After transform, conjugate the result again to complete transpose
    ; (since transform is linear, conjugate(input) -> conjugate(transform(input)))
    mov rcx, rsi
    shr rcx, 1
    mov rbx, rdi
.conjugate_result:
    movsd xmm1, [rbx + 8]
    xorpd xmm1, xmm2
    movsd [rbx + 8], xmm1
    add rbx, 16
    loop .conjugate_result

    jmp .done

.normal_mult:
    call rft_transform_asm

.done:
    
    pop rbp
    ret

;------------------------------------------------------------------------------
; rft_quantum_gate_asm - Apply quantum gate operation
;
; Parameters:
;   rdi - state vector (complex doubles)
;   rsi - gate matrix (complex doubles)
;   rdx - target qubit
;   rcx - control qubit (-1 if none)
;   r8  - number of qubits
;
; Note: Uses the System V AMD64 ABI calling convention
;------------------------------------------------------------------------------
rft_quantum_gate_asm:
    push rbp
    mov rbp, rsp
    
    ; Save parameters
    push r12
    push r13
    push r14
    push r15
    
    ; Calculate state vector size
    mov r10, 1
    mov rcx, r8
    shl r10, cl      ; r10 = 2^(num_qubits)
    
    ; Calculate target bit mask
    mov r11, 1
    mov rcx, rdx
    shl r11, cl      ; r11 = 1 << target_qubit
    
    ; Calculate control bit mask (if applicable)
    mov r12, 0
    cmp rcx, -1
    je .no_control
    mov r12, 1
    mov rcx, rcx
    shl r12, cl      ; r12 = 1 << control_qubit
    
.no_control:
    ; Main computation loop - apply quantum gate
    ; For each basis state |i>, if control condition is met,
    ; apply the gate to the pair |i>, |i ⊕ target_mask>

    xor r13, r13  ; Loop counter i

.gate_loop:
    ; Check if we have a control qubit and if it's set
    test r12, r12
    jz .apply_gate

    ; Check control qubit: (i & control_mask) != 0
    mov r14, r13
    and r14, r12
    jz .skip_gate

.apply_gate:
    ; Calculate paired state: j = i ⊕ target_mask
    mov r14, r13
    xor r14, r11

    ; Only process if j > i to avoid double processing
    cmp r14, r13
    jle .skip_gate

    ; Load amplitudes: amp_i and amp_j
    ; Each complex number is 16 bytes (real double + imag double)
    mov r15, r13
    shl r15, 4      ; i * 16
    movsd xmm0, [rdi + r15]      ; amp_i.real
    movsd xmm1, [rdi + r15 + 8]  ; amp_i.imag

    mov r15, r14
    shl r15, 4      ; j * 16
    movsd xmm2, [rdi + r15]      ; amp_j.real
    movsd xmm3, [rdi + r15 + 8]  ; amp_j.imag

    ; Load gate matrix (2x2 complex = 32 bytes)
    ; Gate format: [[a,b],[c,d]] where each element is complex
    movsd xmm4, [rsi]       ; a.real
    movsd xmm5, [rsi + 8]   ; a.imag
    movsd xmm6, [rsi + 16]  ; b.real
    movsd xmm7, [rsi + 24]  ; b.imag
    movsd xmm8, [rsi + 32]  ; c.real
    movsd xmm9, [rsi + 40]  ; c.imag
    movsd xmm10, [rsi + 48] ; d.real
    movsd xmm11, [rsi + 56] ; d.imag

    ; Apply gate: new_amp_i = a*amp_i + b*amp_j
    ; new_amp_j = c*amp_i + d*amp_j

    ; Compute a*amp_i (complex multiply)
    ; (ar + i*ai) * (br + i*bi) = (ar*br - ai*bi) + i*(ar*bi + ai*br)
    movsd xmm12, xmm4  ; ar
    mulsd xmm12, xmm0  ; ar*br
    movsd xmm13, xmm5  ; ai
    mulsd xmm13, xmm1  ; ai*bi
    subsd xmm12, xmm13 ; ar*br - ai*bi (real part)

    movsd xmm13, xmm4  ; ar
    mulsd xmm13, xmm1  ; ar*bi
    movsd xmm14, xmm5  ; ai
    mulsd xmm14, xmm0  ; ai*br
    addsd xmm13, xmm14 ; ar*bi + ai*br (imag part)

    ; Compute b*amp_j
    movsd xmm14, xmm6  ; br
    mulsd xmm14, xmm2  ; br*cr
    movsd xmm15, xmm7  ; bi
    mulsd xmm15, xmm3  ; bi*ci
    subsd xmm14, xmm15 ; br*cr - bi*ci (real part)

    movsd xmm15, xmm6  ; br
    mulsd xmm15, xmm3  ; br*ci
    movsd xmm0, xmm7   ; bi (reuse xmm0)
    mulsd xmm0, xmm2   ; bi*cr
    addsd xmm15, xmm0  ; br*ci + bi*cr (imag part)

    ; Sum for new_amp_i: (a*amp_i) + (b*amp_j)
    addsd xmm12, xmm14 ; real
    addsd xmm13, xmm15 ; imag

    ; Store new_amp_i
    mov r15, r13
    shl r15, 4
    movsd [rdi + r15], xmm12
    movsd [rdi + r15 + 8], xmm13

    ; Compute c*amp_i
    movsd xmm12, xmm8  ; cr
    mov rax, r13       ; load index
    shl rax, 4         ; multiply by 16
    movsd xmm0, [rdi + rax]     ; reload amp_i.real
    mulsd xmm12, xmm0  ; cr*br
    movsd xmm13, xmm9  ; ci
    movsd xmm1, [rdi + rax + 8] ; reload amp_i.imag
    mulsd xmm13, xmm1  ; ci*bi
    subsd xmm12, xmm13 ; cr*br - ci*bi (real)

    movsd xmm13, xmm8  ; cr
    mulsd xmm13, xmm1  ; cr*bi
    movsd xmm14, xmm9  ; ci
    mulsd xmm14, xmm0  ; ci*br
    addsd xmm13, xmm14 ; cr*bi + ci*br (imag)

    ; Compute d*amp_j
    movsd xmm14, xmm10 ; dr
    mov rbx, r14       ; load index
    shl rbx, 4         ; multiply by 16
    movsd xmm2, [rdi + rbx]     ; reload amp_j.real
    mulsd xmm14, xmm2  ; dr*cr
    movsd xmm15, xmm11 ; di
    movsd xmm3, [rdi + rbx + 8] ; reload amp_j.imag
    mulsd xmm15, xmm3  ; di*ci
    subsd xmm14, xmm15 ; dr*cr - di*ci (real)

    movsd xmm15, xmm10 ; dr
    mulsd xmm15, xmm3  ; dr*ci
    movsd xmm0, xmm11  ; di
    mulsd xmm0, xmm2   ; di*cr
    addsd xmm15, xmm0  ; dr*ci + di*cr (imag)

    ; Sum for new_amp_j: (c*amp_i) + (d*amp_j)
    addsd xmm12, xmm14 ; real
    addsd xmm13, xmm15 ; imag

    ; Store new_amp_j
    mov r15, r14
    shl r15, 4
    movsd [rdi + r15], xmm12
    movsd [rdi + r15 + 8], xmm13

.skip_gate:
    ; Next iteration
    inc r13
    cmp r13, r10
    jl .gate_loop
    
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    ret
