//go:build windows

package main

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"unsafe"
)

// memoryStatusEx mirrors MEMORYSTATUSEX.
type memoryStatusEx struct {
	length               uint32
	memoryLoad           uint32
	totalPhys            uint64
	availPhys            uint64
	totalPageFile        uint64
	availPageFile        uint64
	totalVirtual         uint64
	availVirtual         uint64
	availExtendedVirtual uint64
}

func totalRAMGB() float64 {
	proc := syscall.NewLazyDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx")
	var m memoryStatusEx
	m.length = uint32(unsafe.Sizeof(m))
	if r, _, _ := proc.Call(uintptr(unsafe.Pointer(&m))); r == 0 {
		return 0
	}
	return float64(m.totalPhys) / (1 << 30)
}

func regValue(key, name string) string {
	out, err := exec.Command("reg", "query", key, "/v", name).Output()
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(out), "\n") {
		if strings.Contains(line, name) {
			for _, typ := range []string{"REG_SZ", "REG_DWORD"} {
				if f := strings.SplitN(line, typ, 2); len(f) == 2 {
					return strings.TrimSpace(f[1])
				}
			}
		}
	}
	return ""
}

func cpuName() string {
	return regValue(`HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0`, "ProcessorNameString")
}

// osName reports Windows 11 for builds ≥ 22000 (the registry's ProductName
// still says "Windows 10" on Windows 11).
func osName() string {
	build := regValue(`HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion`, "CurrentBuildNumber")
	n, err := strconv.Atoi(build)
	if err != nil {
		return "Windows"
	}
	if n >= 22000 {
		return fmt.Sprintf("Windows 11 (build %d)", n)
	}
	return fmt.Sprintf("Windows 10 (build %d)", n)
}
