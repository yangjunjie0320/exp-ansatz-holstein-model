import functools, numpy
einsum = functools.partial(numpy.einsum, optimize=True)

def ene(cc_obj, amp):
    # Generated by main.py at 2023-10-01 22:55:06
    # Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    # Hostname:     pauling030
    # Git Branch:   main
    # Git Commit:   05ec14761826cb755a5d22b81e726e76c2cba1c0
    # 
    # amp[0]: a+_v a_o
    # amp[1]: b+
    # amp[2]: b+ a+_v a_o
    # 
    # commu_hbar_order = 1
    # amp_e_order      = 1
    # amp_p_order      = 1
    # bra_e_order      = 0
    # bra_p_order      = 0
    # 
    res  = 1.000000 * einsum('    I,I->',      cc_obj.hpx, amp[1])
    res += 1.000000 * einsum('  ia,ai->',   cc_obj.h1e.ov, amp[0])
    res += 1.000000 * einsum('Iia,Iai->', cc_obj.h1p1e.ov, amp[2])

    # Generated by main.py at 2023-10-01 22:55:10
    # Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    # Hostname:     pauling030
    # Git Branch:   main
    # Git Commit:   05ec14761826cb755a5d22b81e726e76c2cba1c0
    # 
    # amp[0]: a+_v a_o
    # amp[1]: b+
    # amp[2]: b+ a+_v a_o
    # 
    # commu_hbar_order = 2
    # amp_e_order      = 1
    # amp_p_order      = 1
    # bra_e_order      = 0
    # bra_p_order      = 0
    # 
    res += 1.000000 * einsum('Iia,ai,I->', cc_obj.h1p1e.ov, amp[0], amp[1])
    return res


def res_0(cc_obj, amp):
    # Generated by main.py at 2023-10-01 22:55:27
    # Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    # Hostname:     pauling030
    # Git Branch:   main
    # Git Commit:   05ec14761826cb755a5d22b81e726e76c2cba1c0
    # 
    # amp[0]: a+_v a_o
    # amp[1]: b+
    # amp[2]: b+ a+_v a_o
    # res: a+_o a_v
    # 
    # commu_hbar_order = 0
    # amp_e_order      = 1
    # amp_p_order      = 1
    # bra_e_order      = 1
    # bra_p_order      = 0
    # 
    res  = 1.000000 * einsum('ai->ai', cc_obj.h1e.vo)

    # Generated by main.py at 2023-10-01 22:55:30
    # Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    # Hostname:     pauling030
    # Git Branch:   main
    # Git Commit:   05ec14761826cb755a5d22b81e726e76c2cba1c0
    # 
    # amp[0]: a+_v a_o
    # amp[1]: b+
    # amp[2]: b+ a+_v a_o
    # res: a+_o a_v
    # 
    # commu_hbar_order = 1
    # amp_e_order      = 1
    # amp_p_order      = 1
    # bra_e_order      = 1
    # bra_p_order      = 0
    # 
    res +=  1.000000 * einsum('  I,Iai->ai',      cc_obj.hpx, amp[2])
    res += -1.000000 * einsum('  ji,aj->ai',   cc_obj.h1e.oo, amp[0])
    res +=  1.000000 * einsum('  ab,bi->ai',   cc_obj.h1e.vv, amp[0])
    res +=  1.000000 * einsum('  Iai,I->ai', cc_obj.h1p1e.vo, amp[1])
    res += -1.000000 * einsum('Iji,Iaj->ai', cc_obj.h1p1e.oo, amp[2])
    res +=  1.000000 * einsum('Iab,Ibi->ai', cc_obj.h1p1e.vv, amp[2])

    # Generated by main.py at 2023-10-01 22:55:34
    # Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    # Hostname:     pauling030
    # Git Branch:   main
    # Git Commit:   05ec14761826cb755a5d22b81e726e76c2cba1c0
    # 
    # amp[0]: a+_v a_o
    # amp[1]: b+
    # amp[2]: b+ a+_v a_o
    # res: a+_o a_v
    # 
    # commu_hbar_order = 2
    # amp_e_order      = 1
    # amp_p_order      = 1
    # bra_e_order      = 1
    # bra_p_order      = 0
    # 
    res += -1.000000 * einsum('  jb,bi,aj->ai',   cc_obj.h1e.ov, amp[0], amp[0])
    res += -1.000000 * einsum('  Iji,aj,I->ai', cc_obj.h1p1e.oo, amp[0], amp[1])
    res +=  1.000000 * einsum('  Iab,bi,I->ai', cc_obj.h1p1e.vv, amp[0], amp[1])
    res += -1.000000 * einsum('Ijb,aj,Ibi->ai', cc_obj.h1p1e.ov, amp[0], amp[2])
    res += -1.000000 * einsum('Ijb,bi,Iaj->ai', cc_obj.h1p1e.ov, amp[0], amp[2])
    res +=  1.000000 * einsum('Ijb,bj,Iai->ai', cc_obj.h1p1e.ov, amp[0], amp[2])

    # Generated by main.py at 2023-10-01 22:55:38
    # Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    # Hostname:     pauling030
    # Git Branch:   main
    # Git Commit:   05ec14761826cb755a5d22b81e726e76c2cba1c0
    # 
    # amp[0]: a+_v a_o
    # amp[1]: b+
    # amp[2]: b+ a+_v a_o
    # res: a+_o a_v
    # 
    # commu_hbar_order = 3
    # amp_e_order      = 1
    # amp_p_order      = 1
    # bra_e_order      = 1
    # bra_p_order      = 0
    # 
    res += -1.000000 * einsum('Ijb,bi,aj,I->ai', cc_obj.h1p1e.ov, amp[0], amp[0], amp[1])
    return res


def res_1(cc_obj, amp):
    # Generated by main.py at 2023-10-01 22:55:15
    # Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    # Hostname:     pauling030
    # Git Branch:   main
    # Git Commit:   05ec14761826cb755a5d22b81e726e76c2cba1c0
    # 
    # amp[0]: a+_v a_o
    # amp[1]: b+
    # amp[2]: b+ a+_v a_o
    # res: b
    # 
    # commu_hbar_order = 0
    # amp_e_order      = 1
    # amp_p_order      = 1
    # bra_e_order      = 0
    # bra_p_order      = 1
    # 
    res  = 1.000000 * einsum('I->I', cc_obj.hpx)

    # Generated by main.py at 2023-10-01 22:55:18
    # Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    # Hostname:     pauling030
    # Git Branch:   main
    # Git Commit:   05ec14761826cb755a5d22b81e726e76c2cba1c0
    # 
    # amp[0]: a+_v a_o
    # amp[1]: b+
    # amp[2]: b+ a+_v a_o
    # res: b
    # 
    # commu_hbar_order = 1
    # amp_e_order      = 1
    # amp_p_order      = 1
    # bra_e_order      = 0
    # bra_p_order      = 1
    # 
    res += 1.000000 * einsum('  IJ,J->I',      cc_obj.hpp, amp[1])
    res += 1.000000 * einsum('ia,Iai->I',   cc_obj.h1e.ov, amp[2])
    res += 1.000000 * einsum('Iia,ai->I', cc_obj.h1p1e.ov, amp[0])

    # Generated by main.py at 2023-10-01 22:55:22
    # Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    # Hostname:     pauling030
    # Git Branch:   main
    # Git Commit:   05ec14761826cb755a5d22b81e726e76c2cba1c0
    # 
    # amp[0]: a+_v a_o
    # amp[1]: b+
    # amp[2]: b+ a+_v a_o
    # res: b
    # 
    # commu_hbar_order = 2
    # amp_e_order      = 1
    # amp_p_order      = 1
    # bra_e_order      = 0
    # bra_p_order      = 1
    # 
    res += 1.000000 * einsum('Jia,J,Iai->I', cc_obj.h1p1e.ov, amp[1], amp[2])
    return res


def res_2(cc_obj, amp):
    # Generated by main.py at 2023-10-01 22:55:39
    # Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    # Hostname:     pauling030
    # Git Branch:   main
    # Git Commit:   05ec14761826cb755a5d22b81e726e76c2cba1c0
    # 
    # amp[0]: a+_v a_o
    # amp[1]: b+
    # amp[2]: b+ a+_v a_o
    # res: b a+_o a_v
    # 
    # commu_hbar_order = 0
    # amp_e_order      = 1
    # amp_p_order      = 1
    # bra_e_order      = 1
    # bra_p_order      = 1
    # 
    res  = 1.000000 * einsum('Iai->Iai', cc_obj.h1p1e.vo)

    # Generated by main.py at 2023-10-01 22:55:42
    # Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    # Hostname:     pauling030
    # Git Branch:   main
    # Git Commit:   05ec14761826cb755a5d22b81e726e76c2cba1c0
    # 
    # amp[0]: a+_v a_o
    # amp[1]: b+
    # amp[2]: b+ a+_v a_o
    # res: b a+_o a_v
    # 
    # commu_hbar_order = 1
    # amp_e_order      = 1
    # amp_p_order      = 1
    # bra_e_order      = 1
    # bra_p_order      = 1
    # 
    res += -1.000000 * einsum('ji,Iaj->Iai',   cc_obj.h1e.oo, amp[2])
    res +=  1.000000 * einsum('ab,Ibi->Iai',   cc_obj.h1e.vv, amp[2])
    res +=  1.000000 * einsum('IJ,Jai->Iai',      cc_obj.hpp, amp[2])
    res += -1.000000 * einsum('Iji,aj->Iai', cc_obj.h1p1e.oo, amp[0])
    res +=  1.000000 * einsum('Iab,bi->Iai', cc_obj.h1p1e.vv, amp[0])

    # Generated by main.py at 2023-10-01 22:55:46
    # Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    # Hostname:     pauling030
    # Git Branch:   main
    # Git Commit:   05ec14761826cb755a5d22b81e726e76c2cba1c0
    # 
    # amp[0]: a+_v a_o
    # amp[1]: b+
    # amp[2]: b+ a+_v a_o
    # res: b a+_o a_v
    # 
    # commu_hbar_order = 2
    # amp_e_order      = 1
    # amp_p_order      = 1
    # bra_e_order      = 1
    # bra_p_order      = 1
    # 
    res += -1.000000 * einsum('  jb,aj,Ibi->Iai',   cc_obj.h1e.ov, amp[0], amp[2])
    res += -1.000000 * einsum('  jb,bi,Iaj->Iai',   cc_obj.h1e.ov, amp[0], amp[2])
    res += -1.000000 * einsum('  Ijb,bi,aj->Iai', cc_obj.h1p1e.ov, amp[0], amp[0])
    res += -1.000000 * einsum('  Jji,J,Iaj->Iai', cc_obj.h1p1e.oo, amp[1], amp[2])
    res +=  1.000000 * einsum('  Jab,J,Ibi->Iai', cc_obj.h1p1e.vv, amp[1], amp[2])
    res += -1.000000 * einsum('Jjb,Iaj,Jbi->Iai', cc_obj.h1p1e.ov, amp[2], amp[2])
    res += -1.000000 * einsum('Jjb,Ibi,Jaj->Iai', cc_obj.h1p1e.ov, amp[2], amp[2])
    res +=  1.000000 * einsum('Jjb,Ibj,Jai->Iai', cc_obj.h1p1e.ov, amp[2], amp[2])

    # Generated by main.py at 2023-10-01 22:55:50
    # Machine Info: Linux 3.10.0-327.36.3.el7.x86_64
    # Hostname:     pauling030
    # Git Branch:   main
    # Git Commit:   05ec14761826cb755a5d22b81e726e76c2cba1c0
    # 
    # amp[0]: a+_v a_o
    # amp[1]: b+
    # amp[2]: b+ a+_v a_o
    # res: b a+_o a_v
    # 
    # commu_hbar_order = 3
    # amp_e_order      = 1
    # amp_p_order      = 1
    # bra_e_order      = 1
    # bra_p_order      = 1
    # 
    res += -1.000000 * einsum('Jjb,aj,J,Ibi->Iai', cc_obj.h1p1e.ov, amp[0], amp[1], amp[2])
    res += -1.000000 * einsum('Jjb,bi,J,Iaj->Iai', cc_obj.h1p1e.ov, amp[0], amp[1], amp[2])
    return res

