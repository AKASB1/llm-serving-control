package capacity

func Remaining(total, allocated int) int {
	if total < allocated { return 0 }
	return total - allocated
}
